

print('Uniprot from PDB (hello from Anna!)')

def parce_unifile(cdir, fname):

    pdbuni = dict()
    currentID = []
    f = open(cdir+fname)
    for line in f:
        if(line.startswith('//')):
            currentID = []
            continue
        if(line.startswith('AC')):
            lc = line.split()
            strid = line[4:].split(';')
            #print(strid)
            for p in strid:
                pp = p.strip()
                if(len(pp)>0):
                    currentID.append(pp)
            #print(currentID)
        else:
            if(line.startswith('DR   PDB; ')):
                lc = line.split(';')
                pdbid = lc[1].strip()
                #print(pdbid)
                strpdbchains = lc[4].strip().split(',')
                #print(strpdbchains)
                for strpdbchain in strpdbchains:
                    if('=' in strpdbchain):
                        pdbchains = strpdbchain.split('=')[0].split('/')
                        #print(pdbchains)
                        if (len(currentID)>0):
                            print(currentID)
                            strcurrentID = ','.join(currentID)
                            for p in pdbchains:
                                pdbuni[pdbid+p] = strcurrentID
                                # for u in currentID:
                                #     pdbuni[pdbid] = u
                    else:
                        print(currentID)
                        print(line)
                        #exit()
                #print(pdbuni)
    return pdbuni
def annotate_interfaces(cdir, pdbuni):

    wfile = open(cdir+'pdbuni90int_array_chains.txt', 'w')
    ifile = open(cdir+'interface0.9mmrep.txt')
    for l in ifile:
        pp = l.split()
        pdbid1 = pp[0].upper()+pp[1][0:1].upper()
        pdbid2 = pp[0].upper()+pp[1][1:2].upper()
        val1 = '-'
        val2 = '-'
        #print(pdbuni.keys())
        if(pdbid1 in pdbuni.keys()):
            val1 = pdbuni[pdbid1]
        if (pdbid2 in pdbuni.keys()):
            val2 = pdbuni[pdbid2]
        #print(val1)
        #print(val2)
        #exit()
        wfile.write(pp[0]+' '+pp[1]+' '+str(val1)+'*'+str(val2)+'\n')

    ifile.close()
    wfile.flush()
    wfile.close()

def annotate_ecod(cdir, pdbuni):

    wfile = open(cdir+'map_ecod_20000_comparison.txt', 'w')
    ifile = open(cdir+'idx-ecod-pdb2uniprot-raw.txt')
    for l in ifile:
        pp = l.split(',')
        pdbid = pp[1].upper()
        uni1 = pp[2].upper()
        mark = '-'
        uni2 = '-'
        code = pp[3].strip()
        print(code)
        if (pdbid in pdbuni.keys()):
            uni2 = pdbuni[pdbid].upper().strip()
            if(code=='UNP'):
                unps = uni2.split(',')
                mark = '0'
                print(uni1)
                print(unps)
                for unp in unps:
                    print(unp)
                    if(unp == uni1):
                        mark = '1'
                #exit()

        #exit()
        wfile.write(pp[0]+','+pp[1]+','+pp[2]+','+code+','+mark+','+uni2+'\n')

    ifile.close()
    wfile.flush()
    wfile.close()

def main_parce():
    cdir = 'D:/work/bioproteins_dl/data/'
    fname = 'uniprot90interface.txt'
    #wfile = open(cdir+'map_pdb_uni_int90_array_chains.txt', 'w')
    d = parce_unifile(cdir, fname)
    annotate_interfaces(cdir,d)
    # for k, v in d.items():
    #     wfile.write(k.upper()+':'+v.upper()+'\n')
    # wfile.flush()
    # wfile.close()
    return

main_parce()
exit()
