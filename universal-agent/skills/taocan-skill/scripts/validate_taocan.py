"""套餐验证脚本工具。"""
from langchain_core.tools import tool


@tool
def validate_taocan_price(price: str) -> str:
    """验证套餐资费表述是否符合坐席规范。

    Args:
        price: 套餐资费描述（如 "99元/月"、"1188"）
    """
    issues = []
    if '元' not in price:
        issues.append('缺少"元"单位，禁止出现裸数字')
    if any(w in price for w in ['超值', '划算', '便宜']):
        issues.append('包含主观评价词，应使用客观描述')
    if issues:
        return f'不合规: {"; ".join(issues)}'
    return f'合规: "{price}" 符合坐席话术规范'
