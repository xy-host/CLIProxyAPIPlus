package executor

import (
	"bytes"
	"fmt"
	"sort"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type githubCopilotOpenAIResponsesPartState struct {
	Added    bool
	TextDone bool
	PartDone bool
	Text     string
}

type githubCopilotOpenAIResponsesItemState struct {
	Type             string
	CanonicalID      string
	CallID           string
	Name             string
	Added            bool
	Done             bool
	EncryptedContent string
	ReasoningParts   map[int]*githubCopilotOpenAIResponsesPartState
}

type githubCopilotOpenAIResponsesState struct {
	ResponseID                  string
	ItemsByOutputIndex          map[int]*githubCopilotOpenAIResponsesItemState
	RawItemIDToOutputIndex      map[string]int
	CurrentReasoningOutputIndex int
}

func normalizeGitHubCopilotResponsesStreamForOpenAI(line []byte, param *any) [][]byte {
	state := ensureGitHubCopilotOpenAIResponsesState(param)
	if !bytes.HasPrefix(line, dataTag) {
		return [][]byte{line}
	}

	payload := bytes.TrimSpace(line[len(dataTag):])
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) || !gjson.ValidBytes(payload) {
		return [][]byte{line}
	}

	eventType := gjson.GetBytes(payload, "type").String()
	switch eventType {
	case "response.created", "response.in_progress":
		return [][]byte{emitGitHubCopilotResponsesPayload(normalizeGitHubCopilotResponseEnvelopeID(payload, state))}
	case "response.output_item.added":
		return normalizeGitHubCopilotOutputItemAdded(payload, state)
	case "response.reasoning_summary_part.added":
		return normalizeGitHubCopilotReasoningSummaryPartAdded(payload, state)
	case "response.reasoning_summary_text.delta":
		return normalizeGitHubCopilotReasoningSummaryTextDelta(payload, state)
	case "response.reasoning_summary_text.done":
		return normalizeGitHubCopilotReasoningSummaryTextDone(payload, state)
	case "response.reasoning_summary_part.done":
		return normalizeGitHubCopilotReasoningSummaryPartDone(payload, state)
	case "response.output_item.done":
		return normalizeGitHubCopilotOutputItemDone(payload, state)
	case "response.completed", "response.incomplete":
		return normalizeGitHubCopilotCompletedResponse(payload, state)
	case "response.content_part.added", "response.content_part.done", "response.output_text.delta", "response.output_text.done", "response.output_text.annotation.added", "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.code_interpreter_call_code.delta", "response.code_interpreter_call_code.done", "response.image_generation_call.partial_image":
		return [][]byte{emitGitHubCopilotResponsesPayload(normalizeGitHubCopilotItemReferenceEvent(payload, state))}
	default:
		return [][]byte{line}
	}
}

func ensureGitHubCopilotOpenAIResponsesState(param *any) *githubCopilotOpenAIResponsesState {
	if *param == nil {
		*param = &githubCopilotOpenAIResponsesState{
			ItemsByOutputIndex:          make(map[int]*githubCopilotOpenAIResponsesItemState),
			RawItemIDToOutputIndex:      make(map[string]int),
			CurrentReasoningOutputIndex: -1,
		}
	}
	return (*param).(*githubCopilotOpenAIResponsesState)
}

func emitGitHubCopilotResponsesPayload(payload []byte) []byte {
	out := make([]byte, 0, len(dataTag)+2+len(payload))
	out = append(out, dataTag...)
	out = append(out, ' ')
	out = append(out, payload...)
	out = append(out, '\n')
	return out
}

func normalizeGitHubCopilotResponseEnvelopeID(payload []byte, state *githubCopilotOpenAIResponsesState) []byte {
	responseID := gjson.GetBytes(payload, "response.id").String()
	if state.ResponseID == "" && responseID != "" {
		state.ResponseID = responseID
	}
	if state.ResponseID == "" {
		return payload
	}
	updated, err := sjson.SetBytes(payload, "response.id", state.ResponseID)
	if err != nil {
		return payload
	}
	return updated
}

func normalizeGitHubCopilotOutputItemAdded(payload []byte, state *githubCopilotOpenAIResponsesState) [][]byte {
	outputIndex, ok := resolveGitHubCopilotOutputIndex(payload, state, false)
	if !ok {
		return [][]byte{emitGitHubCopilotResponsesPayload(payload)}
	}

	itemType := gjson.GetBytes(payload, "item.type").String()
	rawID := gjson.GetBytes(payload, "item.id").String()
	itemState := ensureGitHubCopilotItemState(state, outputIndex, itemType, rawID)
	itemState.Added = true
	if callID := gjson.GetBytes(payload, "item.call_id").String(); callID != "" {
		itemState.CallID = callID
	}
	if name := gjson.GetBytes(payload, "item.name").String(); name != "" {
		itemState.Name = name
	}
	if enc := gjson.GetBytes(payload, "item.encrypted_content").String(); enc != "" {
		itemState.EncryptedContent = enc
	}
	mergeGitHubCopilotReasoningSummary(itemState, gjson.GetBytes(payload, "item.summary"))

	updated := normalizeGitHubCopilotItemID(payload, itemState.CanonicalID)
	if itemState.Type == "reasoning" {
		if itemState.EncryptedContent != "" {
			updated, _ = sjson.SetBytes(updated, "item.encrypted_content", itemState.EncryptedContent)
		}
		if !gjson.GetBytes(updated, "item.summary").Exists() {
			updated, _ = sjson.SetRawBytes(updated, "item.summary", []byte("[]"))
		}
		state.CurrentReasoningOutputIndex = outputIndex

		results := [][]byte{emitGitHubCopilotResponsesPayload(updated)}
		partState := ensureGitHubCopilotReasoningPartState(itemState, 0)
		if !partState.Added {
			partState.Added = true
			results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryPartAdded(itemState, outputIndex, 0)))
		}
		return results
	}

	return [][]byte{emitGitHubCopilotResponsesPayload(updated)}
}

func normalizeGitHubCopilotReasoningSummaryPartAdded(payload []byte, state *githubCopilotOpenAIResponsesState) [][]byte {
	outputIndex, ok := resolveGitHubCopilotOutputIndex(payload, state, true)
	if !ok {
		return [][]byte{emitGitHubCopilotResponsesPayload(payload)}
	}
	summaryIndex := int(gjson.GetBytes(payload, "summary_index").Int())
	itemState := ensureGitHubCopilotItemState(state, outputIndex, "reasoning", gjson.GetBytes(payload, "item_id").String())
	state.CurrentReasoningOutputIndex = outputIndex

	results := make([][]byte, 0, 2)
	if !itemState.Added {
		itemState.Added = true
		results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningItemAdded(itemState, outputIndex)))
	}

	partState := ensureGitHubCopilotReasoningPartState(itemState, summaryIndex)
	if partState.Added {
		return results
	}
	partState.Added = true
	updated, _ := sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	updated, _ = sjson.SetBytes(updated, "output_index", outputIndex)
	if !gjson.GetBytes(updated, "part.type").Exists() {
		updated, _ = sjson.SetBytes(updated, "part.type", "summary_text")
	}
	results = append(results, emitGitHubCopilotResponsesPayload(updated))
	return results
}

func normalizeGitHubCopilotReasoningSummaryTextDelta(payload []byte, state *githubCopilotOpenAIResponsesState) [][]byte {
	outputIndex, ok := resolveGitHubCopilotOutputIndex(payload, state, true)
	if !ok {
		return [][]byte{emitGitHubCopilotResponsesPayload(payload)}
	}
	summaryIndex := int(gjson.GetBytes(payload, "summary_index").Int())
	itemState := ensureGitHubCopilotItemState(state, outputIndex, "reasoning", gjson.GetBytes(payload, "item_id").String())
	state.CurrentReasoningOutputIndex = outputIndex

	results := make([][]byte, 0, 3)
	if !itemState.Added {
		itemState.Added = true
		results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningItemAdded(itemState, outputIndex)))
	}
	partState := ensureGitHubCopilotReasoningPartState(itemState, summaryIndex)
	if !partState.Added {
		partState.Added = true
		results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryPartAdded(itemState, outputIndex, summaryIndex)))
	}

	delta := gjson.GetBytes(payload, "delta").String()
	partState.Text += delta
	updated, _ := sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	updated, _ = sjson.SetBytes(updated, "output_index", outputIndex)
	results = append(results, emitGitHubCopilotResponsesPayload(updated))
	return results
}

func normalizeGitHubCopilotReasoningSummaryTextDone(payload []byte, state *githubCopilotOpenAIResponsesState) [][]byte {
	outputIndex, ok := resolveGitHubCopilotOutputIndex(payload, state, true)
	if !ok {
		return [][]byte{emitGitHubCopilotResponsesPayload(payload)}
	}
	summaryIndex := int(gjson.GetBytes(payload, "summary_index").Int())
	itemState := ensureGitHubCopilotItemState(state, outputIndex, "reasoning", gjson.GetBytes(payload, "item_id").String())
	partState := ensureGitHubCopilotReasoningPartState(itemState, summaryIndex)
	if partState.Text == "" {
		partState.Text = gjson.GetBytes(payload, "text").String()
	}
	if partState.TextDone {
		return nil
	}
	partState.TextDone = true
	updated, _ := sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	updated, _ = sjson.SetBytes(updated, "output_index", outputIndex)
	if partState.Text != "" {
		updated, _ = sjson.SetBytes(updated, "text", partState.Text)
	}
	return [][]byte{emitGitHubCopilotResponsesPayload(updated)}
}

func normalizeGitHubCopilotReasoningSummaryPartDone(payload []byte, state *githubCopilotOpenAIResponsesState) [][]byte {
	outputIndex, ok := resolveGitHubCopilotOutputIndex(payload, state, true)
	if !ok {
		return [][]byte{emitGitHubCopilotResponsesPayload(payload)}
	}
	summaryIndex := int(gjson.GetBytes(payload, "summary_index").Int())
	itemState := ensureGitHubCopilotItemState(state, outputIndex, "reasoning", gjson.GetBytes(payload, "item_id").String())
	partState := ensureGitHubCopilotReasoningPartState(itemState, summaryIndex)
	if partState.PartDone {
		return nil
	}
	partState.PartDone = true
	updated, _ := sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	updated, _ = sjson.SetBytes(updated, "output_index", outputIndex)
	if partState.Text != "" {
		updated, _ = sjson.SetBytes(updated, "part.text", partState.Text)
	}
	if !gjson.GetBytes(updated, "part.type").Exists() {
		updated, _ = sjson.SetBytes(updated, "part.type", "summary_text")
	}
	return [][]byte{emitGitHubCopilotResponsesPayload(updated)}
}

func normalizeGitHubCopilotOutputItemDone(payload []byte, state *githubCopilotOpenAIResponsesState) [][]byte {
	outputIndex, ok := resolveGitHubCopilotOutputIndex(payload, state, false)
	if !ok {
		return [][]byte{emitGitHubCopilotResponsesPayload(payload)}
	}
	itemType := gjson.GetBytes(payload, "item.type").String()
	itemState := ensureGitHubCopilotItemState(state, outputIndex, itemType, gjson.GetBytes(payload, "item.id").String())
	if callID := gjson.GetBytes(payload, "item.call_id").String(); callID != "" {
		itemState.CallID = callID
	}
	if name := gjson.GetBytes(payload, "item.name").String(); name != "" {
		itemState.Name = name
	}
	if enc := gjson.GetBytes(payload, "item.encrypted_content").String(); enc != "" {
		itemState.EncryptedContent = enc
	}
	mergeGitHubCopilotReasoningSummary(itemState, gjson.GetBytes(payload, "item.summary"))

	results := make([][]byte, 0, 4)
	if itemState.Type == "reasoning" {
		state.CurrentReasoningOutputIndex = outputIndex
		if !itemState.Added {
			itemState.Added = true
			results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningItemAdded(itemState, outputIndex)))
		}
		results = append(results, buildMissingGitHubCopilotReasoningDoneEvents(itemState, outputIndex)...)
	}

	updated := normalizeGitHubCopilotItemID(payload, itemState.CanonicalID)
	if itemState.Type == "reasoning" {
		if itemState.EncryptedContent != "" {
			updated, _ = sjson.SetBytes(updated, "item.encrypted_content", itemState.EncryptedContent)
		}
		updated = setGitHubCopilotReasoningSummary(updated, itemState)
		state.CurrentReasoningOutputIndex = -1
	}
	itemState.Done = true
	results = append(results, emitGitHubCopilotResponsesPayload(updated))
	return results
}

func normalizeGitHubCopilotCompletedResponse(payload []byte, state *githubCopilotOpenAIResponsesState) [][]byte {
	updated := normalizeGitHubCopilotResponseEnvelopeID(payload, state)
	outputs := gjson.GetBytes(updated, "response.output")
	if outputs.IsArray() {
		for i, item := range outputs.Array() {
			itemState := ensureGitHubCopilotItemState(state, i, item.Get("type").String(), item.Get("id").String())
			if enc := item.Get("encrypted_content").String(); enc != "" {
				itemState.EncryptedContent = enc
			}
			mergeGitHubCopilotReasoningSummary(itemState, item.Get("summary"))
		}
	}

	results := make([][]byte, 0, 4)
	for _, outputIndex := range sortedGitHubCopilotOutputIndexes(state) {
		itemState := state.ItemsByOutputIndex[outputIndex]
		if itemState == nil || itemState.Type != "reasoning" || itemState.Done {
			continue
		}
		results = append(results, buildMissingGitHubCopilotReasoningDoneEvents(itemState, outputIndex)...)
		results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningItemDone(itemState, outputIndex)))
		itemState.Done = true
	}

	if outputs.IsArray() {
		for i, item := range outputs.Array() {
			itemState := ensureGitHubCopilotItemState(state, i, item.Get("type").String(), item.Get("id").String())
			updated, _ = sjson.SetBytes(updated, fmt.Sprintf("response.output.%d.id", i), itemState.CanonicalID)
			if itemState.Type == "reasoning" {
				if itemState.EncryptedContent != "" {
					updated, _ = sjson.SetBytes(updated, fmt.Sprintf("response.output.%d.encrypted_content", i), itemState.EncryptedContent)
				}
				updated = setGitHubCopilotReasoningSummaryAtPath(updated, itemState, fmt.Sprintf("response.output.%d.summary", i))
			}
		}
	}
	results = append(results, emitGitHubCopilotResponsesPayload(updated))
	return results
}

func normalizeGitHubCopilotItemReferenceEvent(payload []byte, state *githubCopilotOpenAIResponsesState) []byte {
	outputIndex, ok := resolveGitHubCopilotOutputIndex(payload, state, false)
	if !ok {
		return payload
	}
	rawID := gjson.GetBytes(payload, "item_id").String()
	itemState, ok := state.ItemsByOutputIndex[outputIndex]
	if !ok {
		if rawID == "" {
			return payload
		}
		itemState = ensureGitHubCopilotItemState(state, outputIndex, "", rawID)
	} else if rawID != "" {
		state.RawItemIDToOutputIndex[rawID] = outputIndex
		if itemState.CanonicalID == "" {
			itemState.CanonicalID = rawID
		}
	}
	if itemState.CanonicalID == "" {
		return payload
	}
	updated, err := sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	if err != nil {
		return payload
	}
	return updated
}

func resolveGitHubCopilotOutputIndex(payload []byte, state *githubCopilotOpenAIResponsesState, reasoning bool) (int, bool) {
	if result := gjson.GetBytes(payload, "output_index"); result.Exists() {
		return int(result.Int()), true
	}
	rawID := gjson.GetBytes(payload, "item_id").String()
	if rawID == "" {
		rawID = gjson.GetBytes(payload, "item.id").String()
	}
	if rawID != "" {
		if outputIndex, ok := state.RawItemIDToOutputIndex[rawID]; ok {
			return outputIndex, true
		}
	}
	if reasoning && state.CurrentReasoningOutputIndex >= 0 {
		return state.CurrentReasoningOutputIndex, true
	}
	return 0, false
}

func ensureGitHubCopilotItemState(state *githubCopilotOpenAIResponsesState, outputIndex int, itemType, rawID string) *githubCopilotOpenAIResponsesItemState {
	itemState, ok := state.ItemsByOutputIndex[outputIndex]
	if !ok {
		itemState = &githubCopilotOpenAIResponsesItemState{}
		state.ItemsByOutputIndex[outputIndex] = itemState
	}
	if itemType != "" && itemState.Type == "" {
		itemState.Type = itemType
	}
	if rawID != "" {
		state.RawItemIDToOutputIndex[rawID] = outputIndex
		if itemState.CanonicalID == "" {
			itemState.CanonicalID = rawID
		}
	}
	if itemState.CanonicalID == "" {
		itemState.CanonicalID = buildGitHubCopilotFallbackItemID(state, itemState.Type, outputIndex)
	}
	if itemState.ReasoningParts == nil {
		itemState.ReasoningParts = make(map[int]*githubCopilotOpenAIResponsesPartState)
	}
	return itemState
}

func buildGitHubCopilotFallbackItemID(state *githubCopilotOpenAIResponsesState, itemType string, outputIndex int) string {
	prefix := itemType
	if prefix == "" {
		prefix = "item"
	}
	if state.ResponseID != "" {
		return fmt.Sprintf("%s_%s_%d", prefix, state.ResponseID, outputIndex)
	}
	return fmt.Sprintf("%s_%d", prefix, outputIndex)
}

func ensureGitHubCopilotReasoningPartState(itemState *githubCopilotOpenAIResponsesItemState, summaryIndex int) *githubCopilotOpenAIResponsesPartState {
	if itemState.ReasoningParts == nil {
		itemState.ReasoningParts = make(map[int]*githubCopilotOpenAIResponsesPartState)
	}
	partState, ok := itemState.ReasoningParts[summaryIndex]
	if !ok {
		partState = &githubCopilotOpenAIResponsesPartState{}
		itemState.ReasoningParts[summaryIndex] = partState
	}
	return partState
}

func normalizeGitHubCopilotItemID(payload []byte, canonicalID string) []byte {
	updated, err := sjson.SetBytes(payload, "item.id", canonicalID)
	if err != nil {
		return payload
	}
	return updated
}

func buildMissingGitHubCopilotReasoningDoneEvents(itemState *githubCopilotOpenAIResponsesItemState, outputIndex int) [][]byte {
	results := make([][]byte, 0, 4)
	for _, summaryIndex := range sortedGitHubCopilotReasoningIndexes(itemState) {
		partState := ensureGitHubCopilotReasoningPartState(itemState, summaryIndex)
		if !partState.Added {
			partState.Added = true
			results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryPartAdded(itemState, outputIndex, summaryIndex)))
		}
		if !partState.TextDone {
			partState.TextDone = true
			results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryTextDone(itemState, outputIndex, summaryIndex, partState.Text)))
		}
		if !partState.PartDone {
			partState.PartDone = true
			results = append(results, emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryPartDone(itemState, outputIndex, summaryIndex, partState.Text)))
		}
	}
	if len(results) == 0 {
		partState := ensureGitHubCopilotReasoningPartState(itemState, 0)
		partState.Added = true
		partState.TextDone = true
		partState.PartDone = true
		results = append(results,
			emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryPartAdded(itemState, outputIndex, 0)),
			emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryTextDone(itemState, outputIndex, 0, "")),
			emitGitHubCopilotResponsesPayload(buildGitHubCopilotReasoningSummaryPartDone(itemState, outputIndex, 0, "")),
		)
	}
	return results
}

func buildGitHubCopilotReasoningItemAdded(itemState *githubCopilotOpenAIResponsesItemState, outputIndex int) []byte {
	payload := []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"","type":"reasoning","summary":[]}}`)
	payload, _ = sjson.SetBytes(payload, "output_index", outputIndex)
	payload, _ = sjson.SetBytes(payload, "item.id", itemState.CanonicalID)
	if itemState.EncryptedContent != "" {
		payload, _ = sjson.SetBytes(payload, "item.encrypted_content", itemState.EncryptedContent)
	}
	return payload
}

func buildGitHubCopilotReasoningSummaryPartAdded(itemState *githubCopilotOpenAIResponsesItemState, outputIndex, summaryIndex int) []byte {
	payload := []byte(`{"type":"response.reasoning_summary_part.added","item_id":"","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}`)
	payload, _ = sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	payload, _ = sjson.SetBytes(payload, "output_index", outputIndex)
	payload, _ = sjson.SetBytes(payload, "summary_index", summaryIndex)
	return payload
}

func buildGitHubCopilotReasoningSummaryTextDone(itemState *githubCopilotOpenAIResponsesItemState, outputIndex, summaryIndex int, text string) []byte {
	payload := []byte(`{"type":"response.reasoning_summary_text.done","item_id":"","output_index":0,"summary_index":0,"text":""}`)
	payload, _ = sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	payload, _ = sjson.SetBytes(payload, "output_index", outputIndex)
	payload, _ = sjson.SetBytes(payload, "summary_index", summaryIndex)
	payload, _ = sjson.SetBytes(payload, "text", text)
	return payload
}

func buildGitHubCopilotReasoningSummaryPartDone(itemState *githubCopilotOpenAIResponsesItemState, outputIndex, summaryIndex int, text string) []byte {
	payload := []byte(`{"type":"response.reasoning_summary_part.done","item_id":"","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}`)
	payload, _ = sjson.SetBytes(payload, "item_id", itemState.CanonicalID)
	payload, _ = sjson.SetBytes(payload, "output_index", outputIndex)
	payload, _ = sjson.SetBytes(payload, "summary_index", summaryIndex)
	payload, _ = sjson.SetBytes(payload, "part.text", text)
	return payload
}

func buildGitHubCopilotReasoningItemDone(itemState *githubCopilotOpenAIResponsesItemState, outputIndex int) []byte {
	payload := []byte(`{"type":"response.output_item.done","output_index":0,"item":{"id":"","type":"reasoning","summary":[]}}`)
	payload, _ = sjson.SetBytes(payload, "output_index", outputIndex)
	payload, _ = sjson.SetBytes(payload, "item.id", itemState.CanonicalID)
	if itemState.EncryptedContent != "" {
		payload, _ = sjson.SetBytes(payload, "item.encrypted_content", itemState.EncryptedContent)
	}
	return setGitHubCopilotReasoningSummary(payload, itemState)
}

func setGitHubCopilotReasoningSummary(payload []byte, itemState *githubCopilotOpenAIResponsesItemState) []byte {
	return setGitHubCopilotReasoningSummaryAtPath(payload, itemState, "item.summary")
}

func setGitHubCopilotReasoningSummaryAtPath(payload []byte, itemState *githubCopilotOpenAIResponsesItemState, path string) []byte {
	summary := buildGitHubCopilotReasoningSummary(itemState)
	updated, err := sjson.SetRawBytes(payload, path, summary)
	if err != nil {
		return payload
	}
	return updated
}

func buildGitHubCopilotReasoningSummary(itemState *githubCopilotOpenAIResponsesItemState) []byte {
	parts := sortedGitHubCopilotReasoningIndexes(itemState)
	if len(parts) == 0 {
		return []byte("[]")
	}
	summary := []byte(`[]`)
	for _, summaryIndex := range parts {
		partState := ensureGitHubCopilotReasoningPartState(itemState, summaryIndex)
		entry := []byte(`{"type":"summary_text","text":""}`)
		entry, _ = sjson.SetBytes(entry, "text", partState.Text)
		summary, _ = sjson.SetRawBytes(summary, "-1", entry)
	}
	return summary
}

func mergeGitHubCopilotReasoningSummary(itemState *githubCopilotOpenAIResponsesItemState, summary gjson.Result) {
	if itemState == nil || itemState.Type != "reasoning" || !summary.Exists() || !summary.IsArray() {
		return
	}
	for idx, part := range summary.Array() {
		partState := ensureGitHubCopilotReasoningPartState(itemState, idx)
		partState.Added = true
		text := part.Get("text").String()
		if text != "" {
			partState.Text = text
		}
	}
}

func sortedGitHubCopilotReasoningIndexes(itemState *githubCopilotOpenAIResponsesItemState) []int {
	if len(itemState.ReasoningParts) == 0 {
		return nil
	}
	indexes := make([]int, 0, len(itemState.ReasoningParts))
	for summaryIndex := range itemState.ReasoningParts {
		indexes = append(indexes, summaryIndex)
	}
	sort.Ints(indexes)
	return indexes
}

func sortedGitHubCopilotOutputIndexes(state *githubCopilotOpenAIResponsesState) []int {
	if len(state.ItemsByOutputIndex) == 0 {
		return nil
	}
	indexes := make([]int, 0, len(state.ItemsByOutputIndex))
	for outputIndex := range state.ItemsByOutputIndex {
		indexes = append(indexes, outputIndex)
	}
	sort.Ints(indexes)
	return indexes
}
