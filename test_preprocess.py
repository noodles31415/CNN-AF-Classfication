import Preprocess as pp


if __name__ == "__main__":
    test = './sample2017/validation'
    train = './training2017'
    _, _, _, _ = pp.PrepareFeatures(train, test)
    pp.get_MedAmpInput() #okay
    pp.get_TempInput() #okay
    pp.get_SpecgInput() #okay
 