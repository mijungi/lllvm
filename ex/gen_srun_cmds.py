import os
import sys

""" 
A python script to generate a list of shell commands for submitting jobs 
to a slurm cluster.
@author Wittawat
"""

def gen_srun_ex5():
    """Sequence of srun commands for ex2 (USHCN climate data)"""

    #dataNames = ['tavg_2013']
    #dataNames = ['tavg_1314']
    #dataNames = ['tavg_10y_t14']
    dataNames = ['prcp_10y_t14']
    #dataNames = ['tpravg_f05t14_nor']
    #dataNames = ['prcp_1314', 'prcp_1314_nor']
    # 10-year average of tmin, tmax, precipitation. No tavg.
    #dataNames = ['tmmpr_f05t14_nor', 'prcp_10y_t14', 'tpravg_f05t14_nor']
    #dataNames = ['tmmpr_f05t14_nor']
    Seeds = range(1, 10+1)
    rerun = False
    n = 609
    #Seeds = [7]
    #K = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200 ]
    #K = [11, 12, 13]
    K = [ 3, 6, 9 ]
    #K = [4, 5, 6, 7, 8]
    #K = [ 50, 100, 200, 400 ]
    root_folder = os.path.abspath("../")
    print "#!/bin/bash"
    for dn in dataNames:
        for k in K:
            for s in Seeds:
                fname = 'ushcn-d%s-s%d-k%d-n%d.mat'%(dn, s, k, n)
                fpath = os.path.join(root_folder, 'saved', 'ex5', fname)
                if not rerun and os.path.isfile(fpath):
                    # file exists. Don't generate an srun command.
                    continue
                func = "ex5_ushcn('%s', %d, %d)"%(dn, s, k)
                job_name = "ex5(%s,%d,%d)"%(dn, s, k)

                #cmd = "srun --partition=wrkstn -o /dev/null --qos=short -J \"%s\" matlab -nodesktop -nosplash -singleCompThread -r \"cd %s; startup; %s; exit;\" & "%(job_name, root_folder, func)
                #cmd = "srun  -o /dev/null --qos=short -J \"%s\" matlab -nodesktop -nosplash -singleCompThread -r \"cd %s; startup; %s; exit;\" & "%(job_name, root_folder, func)
                cmd = "matlab -nodesktop -nosplash -singleCompThread -r \"cd %s; startup; %s; exit;\" "%( root_folder, func)
                print cmd



def main():
    if len(sys.argv) < 2:
        print "usage: %s experiment_number" % sys.argv[0]
        sys.exit(1);

    ex_num = int(sys.argv[1])
    func_name = 'gen_srun_ex%d'%ex_num
    eval('%s()'%func_name)
    


if __name__ == '__main__':
    main()
