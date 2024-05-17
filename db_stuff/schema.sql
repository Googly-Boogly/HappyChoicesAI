-- Create the database if it doesn't already exist


-- Create the historical_examples table
CREATE TABLE historical_examples (
    id INT AUTO_INCREMENT PRIMARY KEY,
    situation TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    reasoning TEXT NOT NULL
);