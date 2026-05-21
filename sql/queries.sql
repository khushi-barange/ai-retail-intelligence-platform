-- ============================================================
-- RETAILX — SQL ANALYTICS LAYER
-- All queries run against the retailx_db PostgreSQL database
-- ============================================================


-- ============================================================
-- 1. REVENUE BY CATEGORY
-- Business question: Which product categories make the most money?
-- ============================================================

SELECT
    p.product_category_name_english        AS category,
    COUNT(DISTINCT o.order_id)             AS total_orders,
    ROUND(SUM(o.total_revenue)::NUMERIC, 2) AS total_revenue,
    ROUND(AVG(o.total_revenue)::NUMERIC, 2) AS avg_order_value,
    ROUND(SUM(o.profit)::NUMERIC, 2)        AS total_profit
FROM orders_fact o
JOIN products p
    ON o.order_id IN (
        SELECT order_id FROM orders_fact
    )
WHERE o.order_status = 'delivered'
GROUP BY p.product_category_name_english
ORDER BY total_revenue DESC
LIMIT 15;


-- ============================================================
-- 2. MONTHLY REVENUE TREND
-- Business question: Is revenue growing or declining over time?
-- ============================================================

SELECT
    DATE_TRUNC('month', order_purchase_timestamp) AS month,
    COUNT(DISTINCT order_id)                       AS total_orders,
    ROUND(SUM(total_revenue)::NUMERIC, 2)          AS monthly_revenue,
    ROUND(SUM(profit)::NUMERIC, 2)                 AS monthly_profit,
    ROUND(AVG(total_revenue)::NUMERIC, 2)          AS avg_order_value
FROM orders_fact
WHERE order_status = 'delivered'
  AND order_purchase_timestamp >= '2017-01-01'
GROUP BY DATE_TRUNC('month', order_purchase_timestamp)
ORDER BY month;


-- ============================================================
-- 3. MONTH OVER MONTH REVENUE GROWTH
-- Business question: What is the growth rate each month?
-- Uses window function: LAG()
-- ============================================================

WITH monthly AS (
    SELECT
        DATE_TRUNC('month', order_purchase_timestamp) AS month,
        ROUND(SUM(total_revenue)::NUMERIC, 2)          AS revenue
    FROM orders_fact
    WHERE order_status = 'delivered'
    GROUP BY DATE_TRUNC('month', order_purchase_timestamp)
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month)  AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100
    , 2) AS growth_pct
FROM monthly
ORDER BY month;


-- ============================================================
-- 4. TOP 10 CUSTOMERS BY LIFETIME VALUE
-- Business question: Who are our most valuable customers?
-- ============================================================

SELECT
    customer_unique_id,
    total_orders,
    ROUND(total_spent::NUMERIC, 2)        AS total_spent,
    ROUND(avg_order_value::NUMERIC, 2)    AS avg_order_value,
    ROUND(avg_review_score::NUMERIC, 2)   AS avg_review_score,
    recency_days,
    CASE
        WHEN is_churned = 0 THEN 'Active'
        ELSE 'Churned'
    END AS status
FROM customer_features
ORDER BY total_spent DESC
LIMIT 10;


-- ============================================================
-- 5. CUSTOMER SEGMENTATION (RFM TIERS)
-- Business question: How do we segment customers by value?
-- ============================================================

SELECT
    CASE
        WHEN total_orders >= 5  THEN 'Champion'
        WHEN total_orders >= 3  THEN 'Loyal'
        WHEN total_orders >= 2  THEN 'Returning'
        ELSE                         'One-time'
    END AS customer_segment,
    COUNT(*)                              AS customer_count,
    ROUND(AVG(total_spent)::NUMERIC, 2)   AS avg_lifetime_value,
    ROUND(AVG(recency_days)::NUMERIC, 0)  AS avg_recency_days,
    ROUND(AVG(avg_review_score)::NUMERIC, 2) AS avg_satisfaction
FROM customer_features
GROUP BY customer_segment
ORDER BY avg_lifetime_value DESC;


-- ============================================================
-- 6. REPEAT PURCHASE RATE
-- Business question: What percentage of customers buy more than once?
-- ============================================================

SELECT
    COUNT(*)                                              AS total_customers,
    SUM(CASE WHEN total_orders > 1 THEN 1 ELSE 0 END)    AS repeat_customers,
    ROUND(
        SUM(CASE WHEN total_orders > 1 THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 100
    , 2)                                                  AS repeat_rate_pct,
    ROUND(AVG(total_orders)::NUMERIC, 2)                  AS avg_orders_per_customer
FROM customer_features;


-- ============================================================
-- 7. SHIPPING DELAY ANALYSIS BY STATE
-- Business question: Which states have the worst delivery performance?
-- ============================================================

SELECT
    customer_state,
    COUNT(order_id)                               AS total_orders,
    ROUND(AVG(delivery_days)::NUMERIC, 1)         AS avg_delivery_days,
    SUM(is_late)                                  AS late_deliveries,
    ROUND(
        SUM(is_late)::NUMERIC / COUNT(order_id) * 100
    , 1)                                          AS late_rate_pct,
    ROUND(AVG(
        CASE WHEN is_late = 1 THEN delivery_days END
    )::NUMERIC, 1)                                AS avg_late_delivery_days
FROM orders_fact
WHERE order_status = 'delivered'
  AND delivery_days IS NOT NULL
GROUP BY customer_state
ORDER BY late_rate_pct DESC
LIMIT 15;


-- ============================================================
-- 8. PAYMENT TYPE ANALYSIS
-- Business question: How do customers prefer to pay?
-- ============================================================

SELECT
    payment_type,
    COUNT(*)                                        AS total_orders,
    ROUND(SUM(total_revenue)::NUMERIC, 2)           AS total_revenue,
    ROUND(AVG(payment_installments)::NUMERIC, 1)    AS avg_installments,
    ROUND(AVG(total_revenue)::NUMERIC, 2)           AS avg_order_value
FROM orders_fact
WHERE payment_type IS NOT NULL
GROUP BY payment_type
ORDER BY total_orders DESC;


-- ============================================================
-- 9. REVIEW SCORE DISTRIBUTION
-- Business question: How satisfied are our customers overall?
-- ============================================================

SELECT
    review_score,
    COUNT(*)                                        AS total_reviews,
    ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER () * 100, 1) AS pct_of_total
FROM orders_fact
WHERE review_score IS NOT NULL
GROUP BY review_score
ORDER BY review_score DESC;


-- ============================================================
-- 10. COHORT ANALYSIS — REVENUE BY CUSTOMER ACQUISITION MONTH
-- Business question: Do customers acquired in certain months spend more?
-- Uses window function: RANK()
-- ============================================================

WITH cohorts AS (
    SELECT
        customer_unique_id,
        DATE_TRUNC('month', first_purchase)   AS cohort_month,
        total_spent,
        total_orders
    FROM customer_features
)
SELECT
    cohort_month,
    COUNT(customer_unique_id)                  AS cohort_size,
    ROUND(AVG(total_spent)::NUMERIC, 2)        AS avg_lifetime_value,
    ROUND(AVG(total_orders)::NUMERIC, 2)       AS avg_orders,
    RANK() OVER (ORDER BY AVG(total_spent) DESC) AS value_rank
FROM cohorts
GROUP BY cohort_month
ORDER BY cohort_month;


-- ============================================================
-- 11. SELLER PERFORMANCE
-- Business question: Which sellers drive the most revenue?
-- Uses window function: DENSE_RANK()
-- ============================================================

WITH seller_stats AS (
    SELECT
        oi.seller_id,
        s.seller_state,
        COUNT(DISTINCT oi.order_id)             AS total_orders,
        ROUND(SUM(oi.price)::NUMERIC, 2)        AS total_revenue,
        ROUND(AVG(oi.price)::NUMERIC, 2)        AS avg_item_price,
        ROUND(AVG(r.review_score)::NUMERIC, 2)  AS avg_review_score
    FROM orders_fact o
    JOIN reviews r ON o.order_id = r.order_id
    CROSS JOIN (SELECT order_id, seller_id, price FROM orders_fact LIMIT 1) oi
    JOIN sellers s ON oi.seller_id = s.seller_id
    GROUP BY oi.seller_id, s.seller_state
)
SELECT
    seller_id,
    seller_state,
    total_orders,
    total_revenue,
    avg_item_price,
    avg_review_score,
    DENSE_RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
FROM seller_stats
ORDER BY revenue_rank
LIMIT 10;