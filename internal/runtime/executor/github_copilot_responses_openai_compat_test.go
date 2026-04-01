package executor

import (
	"bytes"
	"testing"

	"github.com/tidwall/gjson"
)

func TestNormalizeGitHubCopilotResponsesStreamForOpenAI_NormalizesReasoningLifecycle(t *testing.T) {
	t.Parallel()

	var param any

	created := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.created","response":{"id":"resp_a","created_at":1,"model":"gpt-5.4"}}`), &param)
	if got := parseGitHubCopilotTestPayload(t, created[0]).Get("response.id").String(); got != "resp_a" {
		t.Fatalf("response.created id = %q, want resp_a", got)
	}

	inProgress := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.in_progress","response":{"id":"resp_b"}}`), &param)
	if got := parseGitHubCopilotTestPayload(t, inProgress[0]).Get("response.id").String(); got != "resp_a" {
		t.Fatalf("response.in_progress id = %q, want resp_a", got)
	}

	added := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_added","type":"reasoning","summary":[]}}`), &param)
	if len(added) != 2 {
		t.Fatalf("reasoning added chunk count = %d, want 2", len(added))
	}
	if got := parseGitHubCopilotTestPayload(t, added[0]).Get("item.id").String(); got != "rs_added" {
		t.Fatalf("reasoning added id = %q, want rs_added", got)
	}
	if got := parseGitHubCopilotTestPayload(t, added[1]).Get("item_id").String(); got != "rs_added" {
		t.Fatalf("reasoning part added item_id = %q, want rs_added", got)
	}

	delta := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_rotated","summary_index":0,"delta":"plan"}`), &param)
	if len(delta) != 1 {
		t.Fatalf("reasoning delta chunk count = %d, want 1", len(delta))
	}
	deltaPayload := parseGitHubCopilotTestPayload(t, delta[0])
	if got := deltaPayload.Get("item_id").String(); got != "rs_added" {
		t.Fatalf("reasoning delta item_id = %q, want rs_added", got)
	}
	if got := deltaPayload.Get("output_index").Int(); got != 0 {
		t.Fatalf("reasoning delta output_index = %d, want 0", got)
	}

	done := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_done","type":"reasoning","summary":[]}}`), &param)
	if len(done) != 3 {
		t.Fatalf("reasoning done chunk count = %d, want 3", len(done))
	}
	if got := parseGitHubCopilotTestPayload(t, done[0]).Get("type").String(); got != "response.reasoning_summary_text.done" {
		t.Fatalf("first done event = %q, want response.reasoning_summary_text.done", got)
	}
	if got := parseGitHubCopilotTestPayload(t, done[0]).Get("item_id").String(); got != "rs_added" {
		t.Fatalf("reasoning text.done item_id = %q, want rs_added", got)
	}
	if got := parseGitHubCopilotTestPayload(t, done[0]).Get("text").String(); got != "plan" {
		t.Fatalf("reasoning text.done text = %q, want plan", got)
	}
	if got := parseGitHubCopilotTestPayload(t, done[2]).Get("item.id").String(); got != "rs_added" {
		t.Fatalf("reasoning output_item.done id = %q, want rs_added", got)
	}

	completed := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.completed","response":{"id":"resp_z","output":[{"id":"rs_final","type":"reasoning","summary":[]}],"usage":{"input_tokens":1,"output_tokens":2}}}`), &param)
	if len(completed) != 1 {
		t.Fatalf("completed chunk count = %d, want 1", len(completed))
	}
	completedPayload := parseGitHubCopilotTestPayload(t, completed[0])
	if got := completedPayload.Get("response.id").String(); got != "resp_a" {
		t.Fatalf("response.completed id = %q, want resp_a", got)
	}
	if got := completedPayload.Get("response.output.0.id").String(); got != "rs_added" {
		t.Fatalf("response.completed reasoning id = %q, want rs_added", got)
	}
	if got := completedPayload.Get("response.output.0.summary.0.text").String(); got != "plan" {
		t.Fatalf("response.completed reasoning summary text = %q, want plan", got)
	}
}

func TestNormalizeGitHubCopilotResponsesStreamForOpenAI_RewritesMessageItemReferences(t *testing.T) {
	t.Parallel()

	var param any

	_ = normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.created","response":{"id":"resp_msg","created_at":1,"model":"gpt-5.4"}}`), &param)
	added := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"msg_added","type":"message","status":"in_progress","content":[],"role":"assistant"}}`), &param)
	if got := parseGitHubCopilotTestPayload(t, added[0]).Get("item.id").String(); got != "msg_added" {
		t.Fatalf("message added id = %q, want msg_added", got)
	}

	partAdded := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.content_part.added","output_index":1,"item_id":"msg_rotated_1","content_index":0,"part":{"type":"output_text","text":""}}`), &param)
	if got := parseGitHubCopilotTestPayload(t, partAdded[0]).Get("item_id").String(); got != "msg_added" {
		t.Fatalf("content_part.added item_id = %q, want msg_added", got)
	}

	textDelta := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.output_text.delta","output_index":1,"item_id":"msg_rotated_2","content_index":0,"delta":"hi"}`), &param)
	if got := parseGitHubCopilotTestPayload(t, textDelta[0]).Get("item_id").String(); got != "msg_added" {
		t.Fatalf("output_text.delta item_id = %q, want msg_added", got)
	}
}

func TestNormalizeGitHubCopilotResponsesStreamForOpenAI_PreservesTerminalReasoningSummary(t *testing.T) {
	t.Parallel()

	var param any

	_ = normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.created","response":{"id":"resp_term","created_at":1,"model":"gpt-5.4"}}`), &param)
	_ = normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_term","type":"reasoning","summary":[]}}`), &param)

	completed := normalizeGitHubCopilotResponsesStreamForOpenAI([]byte(`data: {"type":"response.completed","response":{"id":"resp_rotated","output":[{"id":"rs_rotated","type":"reasoning","summary":[{"type":"summary_text","text":"final summary"}]}],"usage":{"input_tokens":1,"output_tokens":2}}}`), &param)
	if len(completed) != 4 {
		t.Fatalf("completed chunk count = %d, want 4 (synthetic done lifecycle + completed)", len(completed))
	}
	if got := parseGitHubCopilotTestPayload(t, completed[0]).Get("text").String(); got != "final summary" {
		t.Fatalf("synthetic text.done text = %q, want final summary", got)
	}
	if got := parseGitHubCopilotTestPayload(t, completed[3]).Get("response.output.0.summary.0.text").String(); got != "final summary" {
		t.Fatalf("completed reasoning summary text = %q, want final summary", got)
	}
}

func parseGitHubCopilotTestPayload(t *testing.T, chunk []byte) gjson.Result {
	t.Helper()
	if !bytes.HasPrefix(chunk, dataTag) {
		t.Fatalf("chunk missing data prefix: %q", string(chunk))
	}
	payload := bytes.TrimSpace(chunk[len(dataTag):])
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		t.Fatalf("invalid payload: %q", string(payload))
	}
	return gjson.ParseBytes(payload)
}
