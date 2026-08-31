USE shopping_analytics;

-- 01. Monthly revenue
SELECT DATE_FORMAT(o.purchase_at, '%Y-%m') AS revenue_month,
       ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
       COUNT(DISTINCT o.order_id) AS orders
FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY revenue_month ORDER BY revenue_month;

-- 02. Top 10 products by revenue
SELECT p.product_id, p.product_category,
       ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
       COUNT(*) AS item_count
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id
WHERE o.order_status = 'delivered'
GROUP BY p.product_id, p.product_category
ORDER BY revenue DESC LIMIT 10;

-- 03. Top 10 customers by lifetime spend
SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS delivered_orders,
       ROUND(SUM(oi.price + oi.freight_value), 2) AS lifetime_revenue
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_unique_id
ORDER BY lifetime_revenue DESC LIMIT 10;

-- 04. Repeat customer rate
WITH customer_orders AS (
    SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS delivered_orders
    FROM customers c JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT COUNT(*) AS customers_with_orders,
       SUM(delivered_orders >= 2) AS repeat_customers,
       ROUND(100.0 * SUM(delivered_orders >= 2) / COUNT(*), 2) AS repeat_customer_rate_pct
FROM customer_orders;

-- 05. Monthly revenue from new versus existing customers
WITH delivered_orders AS (
    SELECT o.order_id, c.customer_unique_id, o.purchase_at
    FROM orders o JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
),
orders_with_first_month AS (
    SELECT *, MIN(DATE_FORMAT(purchase_at, '%Y-%m')) OVER (
        PARTITION BY customer_unique_id
    ) AS first_purchase_month
    FROM delivered_orders
)
SELECT DATE_FORMAT(o.purchase_at, '%Y-%m') AS revenue_month,
       CASE WHEN DATE_FORMAT(o.purchase_at, '%Y-%m') = o.first_purchase_month THEN 'new'
            ELSE 'existing' END AS customer_type,
       ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue
FROM orders_with_first_month o JOIN order_items oi ON oi.order_id = o.order_id
GROUP BY revenue_month, customer_type
ORDER BY revenue_month, customer_type;

-- 06. Customers without orders. Olist source can return zero rows.
SELECT c.customer_unique_id, c.customer_city, c.customer_state
FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id
WHERE o.order_id IS NULL ORDER BY c.customer_unique_id;

-- 07. Average order value by product category
WITH order_category_revenue AS (
    SELECT o.order_id, p.product_category,
           SUM(oi.price + oi.freight_value) AS category_order_revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    JOIN products p ON p.product_id = oi.product_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, p.product_category
)
SELECT product_category, COUNT(*) AS category_orders,
       ROUND(AVG(category_order_revenue), 2) AS avg_order_value
FROM order_category_revenue
GROUP BY product_category ORDER BY avg_order_value DESC;

-- 08. Dormant customers: no purchase in the 90 days before the latest data date
WITH reference_date AS (
    SELECT MAX(purchase_at) AS max_purchase_at FROM orders WHERE order_status = 'delivered'
),
customer_last_purchase AS (
    SELECT c.customer_unique_id, MAX(o.purchase_at) AS last_purchase_at
    FROM customers c JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT clp.customer_unique_id, clp.last_purchase_at,
       DATEDIFF(rd.max_purchase_at, clp.last_purchase_at) AS days_since_last_purchase
FROM customer_last_purchase clp CROSS JOIN reference_date rd
WHERE clp.last_purchase_at < DATE_SUB(rd.max_purchase_at, INTERVAL 90 DAY)
ORDER BY days_since_last_purchase DESC;

-- 09. Monthly product revenue rank
WITH monthly_product_revenue AS (
    SELECT DATE_FORMAT(o.purchase_at, '%Y-%m') AS revenue_month,
           p.product_id, p.product_category, SUM(oi.price + oi.freight_value) AS revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    JOIN products p ON p.product_id = oi.product_id
    WHERE o.order_status = 'delivered'
    GROUP BY revenue_month, p.product_id, p.product_category
),
ranked_products AS (
    SELECT *, DENSE_RANK() OVER (PARTITION BY revenue_month ORDER BY revenue DESC) AS revenue_rank
    FROM monthly_product_revenue
)
SELECT revenue_month, revenue_rank, product_id, product_category, ROUND(revenue, 2) AS revenue
FROM ranked_products
WHERE revenue_rank <= 3
ORDER BY revenue_month, revenue_rank, product_id;

-- 10. Customer RFM
WITH reference_date AS (
    SELECT MAX(purchase_at) AS max_purchase_at FROM orders WHERE order_status = 'delivered'
),
customer_rfm AS (
    SELECT c.customer_unique_id, MAX(o.purchase_at) AS last_purchase_at,
           COUNT(DISTINCT o.order_id) AS frequency,
           SUM(oi.price + oi.freight_value) AS monetary
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT customer_unique_id,
       DATEDIFF(rd.max_purchase_at, last_purchase_at) AS recency_days,
       frequency, ROUND(monetary, 2) AS monetary
FROM customer_rfm CROSS JOIN reference_date rd
ORDER BY monetary DESC, frequency DESC, recency_days;
