from scale_bert.inference import BERTMultiLabelClassifier

# description = "an asian female with tanned skin and receding hairline without tiny lips.   her hair is medium-length, blond and wavy.  She has blue eyes, pointy nose, arched eyebrows and big ears. She has not large nose. She is putting makeup "
description = "young man. He is fat. His eyes is narrow. His nose is tiny. He doesn't have mustache. He is putting glasses on"
# description = "young man without sunglasses"
bert = BERTMultiLabelClassifier()
logits = bert.predict(description)
print(logits)
