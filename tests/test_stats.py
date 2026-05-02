from tinyadapt.models import BigNet, HalfBigNet
from tinyadapt.utils.stats import profile_model, storage_bytes


def test_fp16_uses_less_storage_than_fp32():
    fp32 = BigNet()
    fp16 = HalfBigNet()

    assert storage_bytes(fp16) < storage_bytes(fp32)


def test_profile_reports_reduction():
    fp32 = BigNet()
    fp16 = HalfBigNet()
    profile = profile_model("fp16", fp16, fp32_storage_bytes=storage_bytes(fp32))

    assert profile.name == "fp16"
    assert profile.reduction_vs_fp32 > 0
