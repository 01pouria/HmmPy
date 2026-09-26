def test_choice_distribution(make_session):
    hmm = make_session([
        '{"value":"A","probabilities":{"A":0.8,"B":0.2}}',
        '{"value":"A","probabilities":{"A":0.6,"B":0.4}}',
    ])
    result = hmm.choice("test", choices=["A", "B"], samples=2)
    assert result.value == "A"
    assert round(result.probabilities["A"], 4) == 0.7
    assert result.usage.requests == 2
