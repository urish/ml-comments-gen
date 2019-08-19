WITH
  stars_table AS (
  -- The following table has been cached as bigtsquery.github.stars
  SELECT
    id repo_id,
    stars,
    repo_name,
    last_star_date,
    first_star_date
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER(PARTITION BY repo_name ORDER BY last_star_date DESC) rn
    FROM (
      SELECT
        repo.id,
        COUNT(DISTINCT actor.id) stars,
        MAX(created_at) last_star_date,
        MIN(created_at) first_star_date,
        ARRAY_AGG(DISTINCT repo.name) repo_names
      FROM
        `githubarchive.month.20*`
      WHERE
        type='WatchEvent'
        AND repo.id IS NOT NULL
      GROUP BY
        1 ),
      UNNEST(repo_names) repo_name )
  WHERE
    rn=1 # same name repos with multiple ids, choose the latest one
    AND repo_name LIKE '%/%' # remove old repo names
    ),
  all_comments AS (
  SELECT
    * EXCEPT (paths),
    CONCAT(SPLIT(path, "/")[
    OFFSET
      (0)], "/", SPLIT(path, "/")[
    OFFSET
      (1)]) AS repo
  FROM
    `bigtsquery.typescript.all_functions_comments`
  JOIN
    UNNEST(paths) AS path
  WHERE
    LENGTH(comments) >= 2 )
SELECT
  id,
  ARRAY_AGG(path) AS paths,
  ANY_VALUE(line) AS line,
  ANY_VALUE(character) AS character,
  ANY_VALUE(comments) AS comments,
  ANY_VALUE(text) AS text,
  MAX((
    SELECT
      stars
    FROM
      stars_table
    WHERE
      repo_name = repo)) AS stars
FROM
  all_comments
GROUP BY
  id