# Honesty data

True and false factual statements from *The Geometry of Truth* (Marks & Tegmark, 2023),
downloaded once and kept unchanged.

- Source: https://github.com/saprmarks/geometry-of-truth, folder `datasets/`
- Commit: `5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb`
- Files (columns `statement,label`, label 1 = true, 0 = false; each half true):

| file | rows | example |
|---|---|---|
| `cities.csv` | 1496 | The city of Krasnodar is in Russia. |
| `neg_cities.csv` | 1496 | The city of Krasnodar is not in Russia. |
| `sp_en_trans.csv` | 354 | The Spanish word 'con' means 'to speak'. |
| `neg_sp_en_trans.csv` | 354 | The Spanish word 'con' does not mean 'to speak'. |
| `larger_than.csv` | 1980 | Fifty-one is larger than fifty-two. |
| `smaller_than.csv` | 1980 | Fifty-one is smaller than fifty-two. |

Used by `traits/honesty.py` (spec §6): statements as plain user messages, with no
goal or role anywhere, for the honesty direction v_honesty.
