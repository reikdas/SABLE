'''
Defined the VBR class to store the VBR matrix data.
'''

class VBR:
    # TODO provide descriptions for each of the parameters
    def __init__(self, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre):
        self.val = val
        self.indx = indx
        self.bindx = bindx
        self.rpntr = rpntr
        self.cpntr = cpntr
        self.bpntrb = bpntrb
        self.bpntre = bpntre
