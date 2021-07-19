from Bert.inference import TextProcessor

# description = "an asian female with tanned skin and receding hairline without tiny lips.   her hair is medium-length, blond and wavy.  She has blue eyes, pointy nose, arched eyebrows and big ears. She has not large nose. She is putting makeup "
# description = "young man. He is fat. His eyes is narrow. His nose is tiny. He doesn't have mustache. He is putting glasses on"
description = "young man with a sunglasses and beard. he has blond hair. His eyes are green"
# description = "young bald man with a beard and glasses"
description = "a man with beard. he has blond hair. His eyes are green"
description= "young bald man with beard and glasses"
# description = "A blonde man with a beard"

bert = TextProcessor('distilbert-base-uncased')
logits = bert.predict(description)
print(logits)

# logits = bert.predict(description)
# # print(logits)

# logits = bert.predict(description)
# print(logits)
