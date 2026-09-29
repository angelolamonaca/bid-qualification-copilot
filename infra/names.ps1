# Resource names used by setup.ps1, deploy.ps1 and cleanup.ps1. Storage and registry names must
# be globally unique: change the suffix if one is taken.
$suffix      = "al26"
$rg          = "rg-bid-copilot"
$location    = "switzerlandnorth"
$aiResource  = "ai-bidcopilot-$suffix"   # created in Step 1
$identity    = "id-bid-copilot"
$storage     = "stbidcopilot$suffix"     # 3-24 lowercase letters and digits
$registry    = "acrbidcopilot$suffix"    # 5-50 letters and digits
$environment = "cae-bid-copilot"
$app         = "bid-copilot"
