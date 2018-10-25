# OpenML-bench-regr
Benchmarking datasets for regression based on OpenML


to-dos:

- Andere Paper/Quellen identifizieren und in der Intro/Literatureteil zitieren (welche haben Klassifikation/Regression/Survival/Clustering Datensätze?)
  - Beispiele für andere Repositories: UCI, OpenML, Kaggle
  - Beispiele für andere benchmarking suites: OpenML100, Penn-ML (Olson et. al), Keel
  - evtl. bei Philipp am Institut? genetische Datensätze?
- zusätzliche Datensätze finden
  - insbesondere big data Datensätze; Sammlung sollte möglichst heterogen sein.
  - z.B. Kaggle Datensätze
  - Lizenzen der einzelnen Datensätze checken und auf OpenML hochladen
- Softwareanbindung: 
  - Auf OpenML taggen (Datensätze oder Tasks?)
  - Bernd kontaktieren und passenden Namen für Tag finden
  - Code schreiben zum "automatischen" Benchmarken auf (einem Teil) der Benchmarking Datensätze
- Bachelorarbeit zu Paper umformen
  - Literatur ergänzen
  - Benchmark verändern? 
    - Evtl. auto tuning Pakete verwenden, wie z.B. tuneRanger oder autoxgboost
    - h2O-AutoML evtl. noch interessant (evtl. andere autoML Algorithmen?)
    - Welche Maße? Kendall, rsq, Pearson, Spearman?
   - kurzer Codeabschnitt mit Beispiel wie man von OpenML die Datensätze runterlädt
- Hier Antwort zu meiner eigenen Frage verfassen, sobald die Datensatzsammlung/das Paper steht: https://opendata.stackexchange.com/questions/12134/regression-datasets-for-benchmarking
