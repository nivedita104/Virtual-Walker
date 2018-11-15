import csv
with open('train_knee_input_v1.csv', 'r') as inp, open('input_knee.csv', 'w',newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        #print(float(row[0]))
        #break
        if float(row[1]) <= 30000.0 and float(row[1]) >= -30000.0 and float(row[4]) <= 30000.0 and float(row[4]) >= -30000.0 and float(row[7]) <= 30000.0 and float(row[7]) >= -30000.0 and float(row[10]) <= 30000.0 and float(row[10]) >= -30000.0 and float(row[13]) <= 30000.0 and float(row[13]) >= -30000.0 and float(row[16]) <= 30000.0 and float(row[16]) >= -30000.0 and float(row[19]) <= 30000.0 and float(row[19]) >= -30000.0 and float(row[22]) <= 30000.0 and float(row[22]) >= -30000.0 and float(row[25]) <= 30000.0 and float(row[25]) >= -30000.0 and float(row[28]) <= 30000.0 and float(row[28]) >= -30000.0 and float(row[31]) <= 30000.0 and float(row[31]) >= -30000.0 and float(row[34]) <= 30000.0 and float(row[34]) >= -30000.0 and float(row[37]) <= 30000.0 and float(row[37]) >= -30000.0 and float(row[40]) <= 30000.0 and float(row[40]) >= -30000.0 and float(row[43]) <= 30000.0 and float(row[43]) >= -30000.0 and float(row[46]) <= 30000.0 and float(row[46]) >= -30000.0 and float(row[49]) <= 30000.0 and float(row[49]) >= -30000.0 and float(row[52]) <= 30000.0 and float(row[52]) >= -30000.0:
            #print(row[0])
            writer.writerow(row)
            #break


