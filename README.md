# OpenML-bench-regr
Benchmarking datasets for regression based on OpenML


to-dos:
- Journal bestimmen. Mögliches Journal: Computational Statistics (Vorschlag ALB)
- Andere Paper/Quellen identifizieren und in der Intro/Literatureteil zitieren (welche haben Klassifikation/Regression/Survival/Clustering Datensätze?)
  - Beispiele für andere Repositories: UCI, OpenML, Kaggle
  - Beispiele für andere benchmarking suites: OpenML100, Penn-ML (Olson et. al), Keel
- zusätzliche Datensätze finden
  - insbesondere big data Datensätze; Sammlung sollte möglichst heterogen sein.
  - z.B. Kaggle Datensätze
  - oder insbesondere hochdimensionale Datensätze, da wir davon noch fast keine haben (mit p > n)
  - evtl. bei Philipp am Institut? genetische Datensätze?
  - Lizenzen der einzelnen Datensätze checken und auf OpenML hochladen
- Softwareanbindung: 
  - Auf OpenML taggen (Datensätze oder Tasks?); Tag oder Study in OpenML? Name?
  - Bernd kontaktieren und passenden Namen für Tag finden
  - Code schreiben zum "automatischen" Benchmarken auf (einem Teil) der Benchmarking Datensätze
- Bachelorarbeit zu Paper umformen
  - Literatur ergänzen
  - kurzer Codeabschnitt mit Beispiel wie man von OpenML die Datensätze runterlädt
  - kurze Beschreibung mit Anzahl Variablen, Features, Tags (z.B. medical, economical,...), etc. 
  - Benchmark verändern und neu rechnen 
    - Seed setzen!
    - Versionen festhalten mit "checkpoint"-Paket!
    - Weitere (mögliche) Pakete für den "Grund"-Benchmark: kknn (statt ibk?), svm (?), ein boosting-algo (?)
    - Evtl. auto tuning Pakete verwenden, wie z.B. tuneRanger oder autoxgboost
    - h2O-AutoML evtl. noch interessant (evtl. andere autoML Algorithmen?)
    - Welche Maße? Kendall, rsq, Pearson, Spearman?
  - keinen festen Datensatz, erweiterbar machen?
  - Name: Regression Suite 2019?
- Hier Antwort zu meiner eigenen Frage verfassen, sobald die Datensatzsammlung/das Paper steht: https://opendata.stackexchange.com/questions/12134/regression-datasets-for-benchmarking
