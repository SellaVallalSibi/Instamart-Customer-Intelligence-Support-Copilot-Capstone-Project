-- Q1: Customers whose completed orders over INR 500 total more than INR 8000
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Q2: Completed revenue by city and product category
SELECT c.city, p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Products priced above the average catalog price
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: Customers without any completed order
SELECT c.customer_id, c.city
FROM customers c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.status = 'completed'
)
ORDER BY c.customer_id;

-- Q5: Customers whose completed-order spend is above the average among active customers
WITH customer_spend AS (
    SELECT o.customer_id, SUM(o.order_amount) AS spend
    FROM orders o
    WHERE o.status = 'completed'
    GROUP BY o.customer_id
)
SELECT customer_id, ROUND(spend, 2) AS spend_inr
FROM customer_spend
WHERE spend > (SELECT AVG(spend) FROM customer_spend)
ORDER BY spend DESC;

-- Q6: Rank customers by completed-order spend within each city
WITH customer_spend AS (
    SELECT c.city, c.customer_id,
           ROUND(SUM(o.order_amount), 2) AS spend
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city, customer_id, spend,
       RANK() OVER (PARTITION BY city ORDER BY spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY spend DESC) AS dense_spend_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY spend DESC, customer_id) AS row_num
FROM customer_spend
ORDER BY city, spend_rank, customer_id;

-- Q7: Daily completed revenue and cumulative running total
WITH daily_revenue AS (
    SELECT substr(order_ts, 1, 10) AS order_date,
           SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT order_date,
       ROUND(daily_revenue, 2) AS daily_revenue_inr,
       ROUND(SUM(daily_revenue) OVER (
           ORDER BY order_date
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ), 2) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY order_date;
