def test_score_numeric(make_session):
    hmm = make_session([
        '{"value":"5","distribution":{"1":0.0,"2":0.0,"3":0.0,"4":0.2,"5":0.8}}',
        '{"value":"4","distribution":{"1":0.0,"2":0.0,"3":0.1,"4":0.4,"5":0.5}}',
    ])
    result = hmm.score("test", scale=[1,2,3,4,5], samples=2)
    assert 4.0 < result.value <= 5.0
    assert result.usage.requests == 2
