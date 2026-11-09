# Business Analyst Documentation

## 1. Role fit
This project fits a **Business Analyst / Data Analyst (technology-focused)** role. The work is problem understanding, data preparation, analysis, KPI definition and recommendations; ML is a tool used to support the decision, not the end goal. It also suits Junior Data Analyst, Business Intelligence Analyst, Risk/Fraud Analyst and Trust & Safety Analyst roles. It is *not* an ML-engineer project (no deployment, MLOps or model tuning at scale).

## 2. Problem statement
Fake and clone accounts damage user trust, spread spam and fraud, and distort engagement metrics. Manual review is slow and cannot cover millions of accounts. The business needs a way to flag likely fake and clone accounts so reviewers can focus on the highest-risk ones.

## 3. Stakeholders
| Stakeholder | Interest |
|---|---|
| Trust & Safety team | Fewer fake accounts, faster review |
| Product / Platform managers | User trust, accurate engagement metrics |
| Brand / Marketing | Protection from impersonation of brand accounts |
| Data / Engineering team | Clear requirements and feature definitions |

## 4. Business requirements (BRD-style)
| ID | Requirement | Priority |
|---|---|---|
| BR-1 | Identify fake accounts from account-level data | Must |
| BR-2 | Identify suspected clone pairs | Must |
| BR-3 | Explain which signals drive the flag (not a black box) | Must |
| BR-4 | Provide KPIs and a dashboard-ready dataset | Should |
| BR-5 | Keep false positives low so genuine users are not wrongly flagged | Must |

## 5. Scope
In scope: analysis of account-level data, prototype detection, KPI reporting.
Out of scope: real-time detection, live Twitter API integration, account suspension workflow.

## 6. Assumptions and constraints
- Dataset labels (bot / genuine) are treated as ground truth.
- Kaggle data is synthetic; findings validate the method only.
- Flagged accounts need human review before any action.

## 7. Success metrics (KPIs)
| KPI | Why it matters |
|---|---|
| Fake account rate (%) | Size of the problem |
| Precision | Of flagged accounts, how many are truly fake (protects genuine users) |
| Recall | Of all fake accounts, how many we catch (protects the platform) |
| F1 score | Balance of both |
| Suspected clone pairs found | Impersonation workload |
| Avg followers / retweets: fake vs genuine | Behaviour gap between groups |

## 8. Analysis workflow (SDLC-style)
1. Understand the problem and stakeholders
2. Define requirements and KPIs
3. Collect data (Kaggle)
4. Clean data and run quality checks
5. Exploratory data analysis
6. Feature engineering
7. Build and compare models
8. Detect clones with Euclidean distance
9. Evaluate against KPIs
10. Present insights and recommendations

## 9. Business insights template (fill with your real numbers)
## 9. Business insights
- Fake accounts make up **46.86%** of the dataset (1,462 out of 3,120 accounts).
- Fake accounts have **fewer** followers on average (**428.8** vs **1,410.7**) but **more** retweets (**37.6** vs **11.9**).
- Usernames with many digits are a strong fake signal (**5.06** digits vs **0.41**).
- The top three signals by importance are: **username digit count**, **follower count**, and **retweet count**.
- **Decision Tree** gave the best F1 (**0.9529**); precision was **0.994** and recall was **0.9151**.
- **30** suspected clone pairs were found; reviewing the top 20 by smallest distance is a practical first batch.


## 10. Recommendations template
1. Add automated flags for accounts with high digit ratio and unusual engagement-per-follower.
2. Route suspected clone pairs to a manual review queue, sorted by distance.
3. Choose the threshold based on cost: favour precision if wrongly blocking users is costly, favour recall if fraud losses are costly.
4. Track the KPIs monthly on a dashboard.
5. Validate on real labelled Twitter data before any rollout.

## 11. Dashboard plan (Power BI / Tableau / Excel)
Load `outputs/dashboard_ready_accounts.csv`, `kpi_summary.csv`, `model_comparison.csv`, `suspected_clone_pairs.csv`.
- **Page 1 Overview:** KPI cards (total accounts, fake count, fake rate), donut of genuine vs fake.
- **Page 2 Behaviour:** bar charts of avg followers/retweets by type, fake rate by location, username digit count distribution.
- **Page 3 Model performance:** table of model metrics, confusion matrix, feature importance bar.
- **Page 4 Clone watchlist:** table of suspected clone pairs with distance and name similarity, filterable.

## 12. Tools used (Business Analyst toolkit)
Python, Pandas, Scikit-learn, Matplotlib/Seaborn, Excel (pivot tables, charts), Power BI or Tableau (dashboard), SQL (optional: load CSV into SQLite/MySQL and re-create KPIs), Git/GitHub (version control), Markdown (documentation), Jira/Confluence style requirement tables (BRD above).

## 13. Business Analyst skills demonstrated
Problem understanding, requirement gathering, stakeholder thinking, structured problem solving, data cleaning and quality checks, exploratory analysis, KPI definition, performance analysis, business insight writing, data storytelling, decision support, documentation, dashboarding.

## 14. Resume bullets
- Analysed a fake and clone Twitter account detection problem and translated it into a structured, data-driven solution with defined requirements and KPIs.
- Prepared and analysed account-level data using Python, Pandas and Scikit-learn to identify behavioural patterns, similarities and potentially fraudulent accounts.
- Applied Euclidean distance with username similarity to compare account profiles and surface suspected clone pairs for review.
- Compared four classification approaches on precision, recall and F1 to support a decision on which flagging approach best balances fraud catch rate and false positives.
- Documented problem statement, requirements, insights and recommendations, and built a dashboard-ready dataset for stakeholder reporting.

## 15. Interview talking points
- *Why precision vs recall?* Wrongly flagging a genuine user hurts trust; missing a fake hurts the platform. The right balance depends on business cost.
- *Why Euclidean distance?* It measures how close two accounts' numeric profiles are; very small distance plus similar usernames suggests a copy.
- *Limitations?* Synthetic data, rule-based clone logic, no real-time pipeline.
