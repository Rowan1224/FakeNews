import re
import pandas as pd

puncList = ["।", "”", "“", "’"]
x = "".join(puncList)
filter= x+'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n০১২৩৪৫৬৭৮৯'


print(filter)

