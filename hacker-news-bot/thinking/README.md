# README

Run the following query on DuckDB and save the results to `data/hacker_news.parquet`.

```sql
SELECT title, url, text, time, timestamp, score AS votes, descendants AS comments
FROM sample_data.hn.hacker_news
WHERE type = 'story'
AND text IS NOT NULL
```
