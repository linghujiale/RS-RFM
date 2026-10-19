# 中文注释：检验交付包核心模块能够正常导入。
from lrfm import __version__
from lrfm.ltc.init import sample_log_uniform
from lrfm.utils.logging import config_hash
from lrfm.utils.seeds import make_seed_bundle


def test_package_version_is_available():
    assert __version__ == "0.1.0"


def test_seed_bundle_is_deterministic():
    first = make_seed_bundle(42).numpy.normal(size=4)
    second = make_seed_bundle(42).numpy.normal(size=4)
    assert first.tolist() == second.tolist()


def test_log_uniform_samples_within_bounds():
    rng = make_seed_bundle(7).numpy
    samples = sample_log_uniform(rng, 0.01, 10.0, size=128)
    assert samples.min() >= 0.01
    assert samples.max() <= 10.0


def test_config_hash_is_stable():
    assert config_hash({"b": 2, "a": 1}) == config_hash({"a": 1, "b": 2})
