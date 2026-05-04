#!/bin/bash
echo "Downloading BEATs Iter3+ (AS2M) Model..."
wget "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmX0T5BxKI1I1s01CQFlLy6ROTrWEIllByrxicmXQ%3D" -O BEATs_iter3_plus_AS2M.pt
echo "Downloaded Successfully to model_zoo/BEATs_iter3_plus_AS2M.pt"
