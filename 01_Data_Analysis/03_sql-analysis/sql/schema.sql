CREATE DATABASE IF NOT EXISTS shopping_analytics
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE shopping_analytics;

CREATE TABLE IF NOT EXISTS customers (
    customer_id CHAR(32) PRIMARY KEY,
    customer_unique_id CHAR(32) NOT NULL,
    customer_city VARCHAR(100) NOT NULL,
    customer_state CHAR(2) NOT NULL,
    INDEX idx_customers_unique_id (customer_unique_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS products (
    product_id CHAR(32) PRIMARY KEY,
    product_category VARCHAR(100) NOT NULL,
    product_weight_g INT NULL,
    product_length_cm INT NULL,
    product_height_cm INT NULL,
    product_width_cm INT NULL,
    INDEX idx_products_category (product_category)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS orders (
    order_id CHAR(32) PRIMARY KEY,
    customer_id CHAR(32) NOT NULL,
    order_status VARCHAR(20) NOT NULL,
    purchase_at DATETIME NOT NULL,
    approved_at DATETIME NULL,
    delivered_carrier_at DATETIME NULL,
    delivered_at DATETIME NULL,
    estimated_delivery_at DATETIME NULL,
    CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    INDEX idx_orders_purchase_at (purchase_at),
    INDEX idx_orders_status_purchase_at (order_status, purchase_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS order_items (
    order_id CHAR(32) NOT NULL,
    order_item_id INT NOT NULL,
    product_id CHAR(32) NOT NULL,
    seller_id CHAR(32) NOT NULL,
    shipping_limit_at DATETIME NOT NULL,
    price DECIMAL(12, 2) NOT NULL,
    freight_value DECIMAL(12, 2) NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    CONSTRAINT fk_order_items_order FOREIGN KEY (order_id) REFERENCES orders(order_id),
    CONSTRAINT fk_order_items_product FOREIGN KEY (product_id) REFERENCES products(product_id),
    INDEX idx_order_items_product (product_id)
) ENGINE=InnoDB;
