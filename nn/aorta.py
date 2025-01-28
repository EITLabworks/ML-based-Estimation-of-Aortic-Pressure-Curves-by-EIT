import numpy as np


class AortaNormalizer:
    def __init__(self, paratype="Linear", mode="fixed", factor=1, deduction=0):
        self.sParatype = paratype
        self.sMode = mode
        self._make_standard_vector(10)
        self.dLengthParas= {"Linear":42, "CauchyLorentz":58, "PolyHierarchicalLin": 43, "Full": 1024}
        if paratype in self.dLengthParas:
            self._make_standard_vector(self.dLengthParas[paratype])
        if self.sMode == "fixed":
            if self.sParatype=="Linear":
                posIndex = np.arange(0,self.iL,2)
                self.pfFactor[posIndex] = 1024
                self.pfFactor[posIndex+1] = 20
                self.pfOffset[posIndex+1] = 85
            elif self.sParatype=="CauchyLorentz":
                self.pfFactor[0] = 1024
                self.pfFactor[1] = 20
                self.pfOffset[1] = 85
                index = np.arange(2, self.iL, 4)
                self.pfFactor[index] = 1024
                self.pfFactor[index+1] = 20000
                self.pfFactor[index+2] = 20000
                self.pfFactor[index+3] = 300
                #evtl. noch mit zero mean arbeiten

            elif self.sParatype=="PolyHierarchicalLin":
                self.pfFactor[0] = 1024
                self.pfFactor[1] = 20
                self.pfOffset[1] = 85
                self.pfFactor[2:7] = 1000
                index= np.arange(7, self.iL, 2)
                self.pfFactor[index] = 1024
                self.pfFactor[index+1] = 4
                #evtl. noch mit zero mean arbeiten
            elif self.sParatype=="Full":
                self.pfFactor[:]= 20
                self.pfOffset[:]= 85
        else:
            self.pfFactor[:] = factor
            self.pfOffset[:] = deduction

        print(f"Init Aorta normalizer with\n Paratype={self.sParatype} and Mode={self.sMode}.")

    def _make_standard_vector(self, L):
        self.iL = L
        self.pfFactor = np.ones(self.iL)
        self.pfOffset = np.zeros(self.iL)

    def normalize_forward(self, ppfAorta):
        return np.divide(np.add(ppfAorta, -1*self.pfOffset),self.pfFactor)

    def normalize_inverse(self, ppfAorta):
        return np.add(np.multiply(ppfAorta, self.pfFactor), self.pfOffset)

    def reset_normalizer(self, L):
        self._make_standard_vector(L)



class AortaParameterHandler:
    def __init__(self, paratype="Linear", mode="fixed", factor=1, deduction=0):
        self.sParatype = paratype
        self.sMode = mode


    def work_forward(self):
        pass
       # resampling, reordering, quality checks, Aortanorm

    def work_inverse(self):
        pass