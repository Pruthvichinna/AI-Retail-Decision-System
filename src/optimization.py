# src/optimization.py

import pulp

def run_promotion_optimization(product_summary):
    weekly_discount_budget = 500.00
    model = pulp.LpProblem("Promotion_Optimization", pulp.LpMaximize)
    products = product_summary['product_id'].tolist()
    promo_vars = pulp.LpVariable.dicts("Promote", products, cat='Binary')

    revenue_expression = pulp.lpSum([
        (1 - promo_vars[p]) * product_summary.loc[i, 'avg_price'] * product_summary.loc[i, 'weekly_demand_forecast'] +
        (promo_vars[p]) * product_summary.loc[i, 'discount_price'] * product_summary.loc[i, 'promo_demand_forecast']
        for i, p in enumerate(products)
    ])
    model += revenue_expression, "Total_Revenue"

    discount_cost_expression = pulp.lpSum([
        promo_vars[p] * (product_summary.loc[i, 'avg_price'] - product_summary.loc[i, 'discount_price']) * product_summary.loc[i, 'promo_demand_forecast']
        for i, p in enumerate(products)
    ])
    model += discount_cost_expression <= weekly_discount_budget, "Total_Discount_Budget_Constraint"

    model.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 suppresses solver output

    # Package the results into a dictionary
    results = {
        'promoted_products': [p for p in products if promo_vars[p].varValue == 1],
        'expected_revenue': pulp.value(model.objective),
        'discount_cost': pulp.value(discount_cost_expression)
    }
    return results