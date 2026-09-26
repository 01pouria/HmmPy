def test_decisions_ignore_history_by_default(make_session):
    hmm = make_session(['hello'])
    hmm.chat("remember this")
    hmm._client = type(hmm._client)(['{"value":true,"yes_probability":0.9}'])
    hmm.boolean("test", samples=1)
    sent = hmm._client.chat.completions.calls[0]["messages"]
    assert all(m["content"] != "remember this" for m in sent)

def test_decisions_can_use_history(make_session):
    hmm = make_session(['hello'])
    hmm.chat("remember this")
    hmm._client = type(hmm._client)(['{"value":true,"yes_probability":0.9}'])
    hmm.boolean("test", samples=1, use_history=True)
    sent = hmm._client.chat.completions.calls[0]["messages"]
    assert any(m["content"] == "remember this" for m in sent)
