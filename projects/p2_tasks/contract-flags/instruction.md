Review the five service agreements in this folder (`msa_*.txt`) against the firm's rules in `playbook.json`.

Write a file named `report.json` in this folder. It must map each contract id (the file name without `.txt`) to a sorted list of the playbook rule names that the contract breaks. Use an empty list when a contract breaks no rules.

Example shape: {"msa_01": [], "msa_05": ["governing_law"]}
