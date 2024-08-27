# CHANGELOG for Semantic-Segmentation

### v3.0.1 - (Nate Haddad, 8/26/2024)
* Add `report.pdf`

### v3.0.0 - (Nate Haddad, 12/17/2022)
* Add `process_video.py` script to process video data
* Refactor models into a `DeepLabWrapper` class
* Update scripts and files

### v2.0.0 - (Nate Haddad, 5/7/2022)
* Switch to yaml configuration files
* Create base `Trainer` class for future use in different training setups
* Major refactoring to repository
* Update README and add CHANGELOG

### v1.0.0 - (Nate Haddad, 5/19/2021)
* Initial commits

## KNOWN ISSUES
* Class indices may not correspond to correct string class names
* Training dataset may have incorrect labels in some images
