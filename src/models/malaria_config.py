# ==============================================================================
# Configuration file that contains any hyperparameters / Feature selection
# ==============================================================================

# Note PVRII (1314 non-null), Gexp.18 (1410 non null), S..Mansoni.GST.control (1410 non-null) has more missing values than others

numeric_feat = ["PvRII","PfSEA","PfEtramp.5.Ag.1","PmMSP119","GST.lshtm", "PfAMA1",
"PfMSP.119", "PkSSP2_x", "PvDBPII", "PvMSP119", "MSP2.CH150.9","MSP2.Dd2", "PvEBP",
"Gexp.18", "PkAMA1","S..mansoni.GST.control", "Tet.Tox.lshtm", "PfGlurp", "Pk.SERA.Ag2","PvAMA1",
"age"
]

# Note "Sample","individualID","mergeID" has been removed as it doesnt make any use for prediction
# We remove houseID as there too many catgories and this affects one hot encoding
cat_feat = ["gender","Bangsa", "occupation", "goForest", "sawMonkey", 'pf','pv']

geo_feat = ["GPS_X","GPS_Y"]

target = ["hadMalaria"]

EPOCHS = 25
BATCH_SIZE = 16 #There isnt many samples in the malaria dataset