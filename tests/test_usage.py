from hmmpy import Pricing, Usage

def test_pricing():
    pricing = Pricing(input=1.0, output=2.0)
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert pricing.estimate(usage) == 2.0
