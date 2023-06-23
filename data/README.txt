|| =========================== ||
|| Description of the datasets ||
|| =========================== ||

- abalone: shape (4177, 9)
Regression (or 28-classes classif), nb_rings in last column
Column 1 is sex 'M' male, 'F' female, 'I' infant (meaningless)
Columns 2~8 are abalone features

- breast: shape (569, 32)
Binary classif, label 'M' (malignant) or 'B' (benign) in column 2
Column 1 is patient ID (meaningless)
Columns 3~32 are tumor data

- gaussians: shape (10000, 10)
3-classes classif, label 'Class0', 'Class1' or 'Class2' in last column
This dataset is mine
Column 1 is row id (meaningless)
Columns 2~9 are data (mixture of 3 Gaussians with k=4 factors and dim=8)

- geyser: shape (272, 4)
Binary classif, label 'long' or 'short' in last column
Columns 1 is geyser ID (meaningless)
Columns 2 and 3 are geyser data

- japanese_vowels: (9961, 15)
9-classes classif, label is '1', '2', ..., '9' in column 1
Columns 2 and 3 are utterances ID (meaningless)
Columns 4~15 are features

- penguin: shape (342, 8)
3-classes classif, 'Adelie', 'Chinstrap', 'Gentoo' in column 2
Column 1 is penguin ID from original dataset (meaningless)
Columns 3~6 are penguin data

- planets: shape (550, 6)
No specific task, this dataset is mine
columns 1, 2, 3, 4 and 6 are in log

- pollen: shape (3848, 6)
No specific task
Columns 1~5 are pollen features: ridge, nub, crack, weight, density
Column 6 is observation number (meaningless)

- sulfur: shape (10081, 7)
No specific task.
Columns 1~7 are numerical features for sulfur recovery units

- sylvine: (5124, 21)
Binary classif, label 0 or 1 in first column
Columns 2~21 are features

-wine_red/wine_white: shape (1599, 12) and (4898, 12)
Regression (or 6-classes classif), quality in last column {3,4,5,6,7,8,9}
Columns 1~11 are wines data
