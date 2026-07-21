class UnitConversion:
    def __init__(self, factor):
        self.factor=factor

    def convert(self,value):
        return value*self.factor



kw_mw=UnitConversion(factor=0.001)
print (kw_mw.convert(250.0))