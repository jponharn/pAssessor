import pandas as pd
import numpy as np
from joblib import dump, load
import re
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import glob
from BuildingCounter import * 
from Counter import Counter
from pythainlp import spell

class CreateModel:

    def __init__(self):
        self.score = {}

    def loadData(self):
        df_placetype = pd.read_csv("./sources/placetype_training_data.csv")
        df_data = df_placetype[df_placetype.classification != 'สถานที่ราชการ']
        df_data = df_placetype.copy()
        df_data.classification = df_data.classification.map({'สถานศึกษา':'education', 'โรงงาน/สถานที่ทำงาน':'factory', 'หมู่บ้าน/ชุมชน':'habitat', 'สาธารณสุข':'health', 'โรงแรม/รีสอร์ท':'hotel', 'ศาสนสถาน':'religious'})
        return df_data

    def cleansing(self, df_data):
        df = pd.read_csv("./sources/manualcleaned6placetype.csv", sep ="|")

        #drop 1 habitat 1 plase
        ###########################################################################################
        df_places_bcount = df.groupby(['place_namt'])['building_name'].count().reset_index()
        p = re.compile('[0-9 ๐-๙ ? !]+/*[0-9]*(ม)*(m)*(ซ)*(ต)*(อ)*(\.)*[0-9]*')
        places_b1 = df_places_bcount[df_places_bcount.building_name == 1]
        cnt = 0
        _1h1c = [] 
        for index, pb1 in places_b1.iterrows():
            if p.match(pb1.place_namt):
                cnt += 1
                _1h1c.append(pb1.place_namt)
        # print("1habitat_1Place with address pattern =", cnt, "places")
        df_data = df_data[~df_data.place_namt.isin(_1h1c)].reset_index(drop=True)
        
        #drop other classified
        ###########################################################################################
        other = df_data[df_data.keyword == 'other']
        other_class = []
        other_columns = ["address", "person", "habitat", "education", "religious", "health", "hotel", "factory", "other"]
        for o in other[other_columns].values:
            data = pd.Series(data=o.tolist(), index=other_columns)
            other_class.append(data.idxmax())
        other['other_class'] = other_class
        other['other_class'] = other.other_class.map({'address':'habitat', 'person':'habitat', 'habitat':'habitat', 'education':'education', 'religious':'religious', 'health':'health', 'hotel':'hotel', 'factory':'factory', 'other':'other'})
        placetype_cant_map = other[other.classification != other.other_class]

        #drop unclassified
        ###########################################################################################
        palacetype_cann_map = df_data[~df_data.place_id.isin(placetype_cant_map.place_id.tolist())]

        #cleansing data
        ###########################################################################################

        #habitat
        habitat_type = palacetype_cann_map[palacetype_cann_map.classification == 'habitat']
        habitat_type_false = habitat_type[(habitat_type.address + habitat_type.person + habitat_type.habitat + habitat_type.otherworkspace + habitat_type.cospace) == 0]
        habitat_type = habitat_type[~habitat_type.place_id.isin(habitat_type_false.place_id)]
        habitat_embig = habitat_type[(habitat_type.education + habitat_type.religious + habitat_type.health + habitat_type.hotel + habitat_type.factory) > 0]
        habitat_type = habitat_type[~habitat_type.place_id.isin(habitat_embig.place_id)]
        habitat_type

        #education
        education_type = palacetype_cann_map[palacetype_cann_map.classification == 'education']
        education_type_false = education_type[(education_type.education + education_type.otherworkspace + education_type.cospace + education_type.government + education_type.other) == 0]
        education_type = education_type[~education_type.place_id.isin(education_type_false.place_id)]
        education_embig = education_type[(education_type.address + education_type.person + education_type.habitat + education_type.person + education_type.religious + education_type.health + education_type.hotel + education_type.factory) > 0]
        education_type = education_type[~education_type.place_id.isin(education_embig.place_id)]
        
        #religious
        religious_type = palacetype_cann_map[palacetype_cann_map.classification == 'religious']
        religious_type_false = religious_type[(religious_type.religious + religious_type.otherworkspace + religious_type.cospace  + religious_type.other) == 0]
        religious_type = religious_type[~religious_type.place_id.isin(religious_type_false.place_id)]
        religious_embig = religious_type[(religious_type.address + religious_type.person + religious_type.habitat + religious_type.person + religious_type.education + religious_type.health + religious_type.hotel + religious_type.factory) > 0]
        religious_type = religious_type[~religious_type.place_id.isin(religious_embig.place_id)]

        #health
        health_type = palacetype_cann_map[palacetype_cann_map.classification == 'health']
        health_type_false = health_type[(health_type.health + health_type.otherworkspace + health_type.cospace + health_type.other) == 0]
        health_type = health_type[~health_type.place_id.isin(health_type_false.place_id)]
        health_embig = health_type[(health_type.address + health_type.person + health_type.habitat + health_type.person + health_type.education + health_type.religious + health_type.hotel + health_type.factory) > 0]
        health_type = health_type[~health_type.place_id.isin(health_embig.place_id)]

        #hotel
        hotel_type = palacetype_cann_map[palacetype_cann_map.classification == 'hotel']
        hotel_type_false = hotel_type[(hotel_type.hotel + hotel_type.otherworkspace + hotel_type.cospace + hotel_type.other) == 0]
        hotel_type = hotel_type[~hotel_type.place_id.isin(hotel_type_false.place_id)]
        hotel_embig = hotel_type[(hotel_type.address + hotel_type.person + hotel_type.habitat + hotel_type.person + hotel_type.education + hotel_type.religious + hotel_type.health + hotel_type.factory) > 0]
        hotel_type = hotel_type[~hotel_type.place_id.isin(hotel_embig.place_id)]

        #factory
        factory_type = palacetype_cann_map[palacetype_cann_map.classification == 'factory']
        factory_type_false = factory_type[(factory_type.factory + factory_type.otherworkspace + factory_type.cospace + factory_type.other) == 0]
        factory_type = factory_type[~factory_type.place_id.isin(factory_type_false.place_id)]
        factory_embig = factory_type[(factory_type.address + factory_type.person + factory_type.habitat + factory_type.person + factory_type.education + factory_type.religious + factory_type.health + factory_type.hotel) > 0]
        factory_type = factory_type[~factory_type.place_id.isin(factory_embig.place_id)]

        #prepair data
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(habitat_type_false.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(habitat_embig.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(habitat_type[(habitat_type.address + habitat_type.person + habitat_type.habitat) == 0].place_id.tolist())]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(habitat_type[(habitat_type.address + habitat_type.person + habitat_type.habitat) < 10].place_id.tolist())]

        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(education_type_false.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(education_embig.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(education_type[education_type.education == 0].place_id.tolist())]

        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(religious_type_false.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(religious_embig.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(religious_type[religious_type.religious == 0].place_id.tolist())]

        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(health_type_false.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(health_embig.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(health_type[health_type.health == 0].place_id.tolist())]

        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(hotel_type_false.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(hotel_embig.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(hotel_type[hotel_type.hotel == 0].place_id.tolist())]

        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(factory_type_false.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(factory_embig.place_id)]
        palacetype_cann_map = palacetype_cann_map[~palacetype_cann_map.place_id.isin(factory_type[factory_type.factory == 0].place_id.tolist())]

        return palacetype_cann_map

    def createLogistic(self, df_data):
        models_path = "./Models"
        modelsList = {'education':0, 'factory':0,'habitat':0, 'health':0, 'hotel':0,'religious':0}

        for model in modelsList.keys():
            models = modelsList.copy()
            data_train = df_data.copy()
            models[model] = 1
            data_train.classification = data_train.classification.map(models)

            calss_1 = data_train[data_train.classification == 1]
            calss_0 = data_train[data_train.classification == 0]

            habitat_upsampled = resample(calss_1, replace=True, n_samples=len(calss_0), random_state=27) 
            upsampled = pd.concat([calss_0, habitat_upsampled])
            upsampled.classification.value_counts()

            X = upsampled[['address', 'person', 'habitat', 'education','religious', 'health', 'hotel', 'factory', 'otherworkspace', 'cospace']]
            y = upsampled['classification']
            save_model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X,y)
            score = model_selection.cross_val_score(save_model, X, y, cv=10, scoring='accuracy')
            self.score[model] = {"accuracy": score.mean(),"std": score.std()}
            dump(save_model, "{}/{}".format(models_path, "lr_{}.sav".format(model)))
        return self.score

class PlaceTypeQuality:

    def __init__(self):
        self.models = {}
        files = glob.glob("Models/lr_*.sav")
        for file in files:
            self.models[file.split('lr_')[1].split('.sav')[0]] = load(file)

    # def predict(self, data):
    #     for model in self.models:
    #         # self.models[model].predict(data)
    #         print('{}-{}'.format(model, self.models[model].predict(data)))

    def placetype_proba(self, data):
        result = {}
        for model in self.models:
            result[model] = self.models[model].predict_proba(data)[:,1]

        raw = {'habitat': result['habitat'].round(5), 'education': result['education'].round(5), 'factory': result['factory'].round(5), 'health': result['health'].round(5), 'hotel': result['hotel'].round(5), 'religious': result['religious'].round(5)}
        return pd.DataFrame(data=raw)

    def proba(self, data, placeType):
        propa = self.placetype_proba(data=data)
        result = []
        for i, place in propa.iterrows():
            result.append(propa[placeType[i]][i])

        return pd.DataFrame(data={'place_type': placeType, 'propa': result})

    def placetype_assess(self, data, coef:float = 0.85):
        result = {}
        proba = self.placetype_proba(data=data)
        result['quality'] = np.sum(proba.values >= coef, axis=1)
        quality = pd.DataFrame(data=result)
        quality['quality_level'] = quality['quality'].map(lambda q: self.__getQuality(q))
        return quality
    
    def __getQuality(self, q):
        if q == 1:
            return "high"
        elif q == 0:
            return "low"
        else:
            return "ambiguous"


class BuildingTransform:
    def __init__(self):
        self.adp = re.compile('[a-zA-Zก-ฮ]+\.+[0-9๐-๙]+')
        self.p = re.compile('[0-9๐-๙ ]{1,4}\/+[0-9๐-๙ ]*[มMซตอ]*\.+[ 0-9๐-๙]*')
        self.h = re.compile('[a-zA-Zก-ฮ ? !]+[0-9]+')
        self.x = re.compile('\w{0,1}[0-9๐-๙]+')
        self.bCount = BuildingCounter()
        self.bc = BuildingClassify()
        self.bw = BuildingWeightChecker()
        self.person_corpus = pd.read_csv("./sources/person_corpus.csv", sep ="|").name.tolist()

    def __correctBuilding(self, building):
        rmList = ['อาคาร', 'ตึก', 'โรง', 'ศูนย์', 'ศูนย', 'ศูนร์', 'โงง', 'โรว', 'อาคาน', 'หอ', 'มหา', 'งาน', 'ศาลา']
        words = []
        pref = ""
        if building.startswith(tuple(rmList)):
            for rm in rmList:
                if building.startswith(rm):
                    pref = rm
                    building = building.replace(rm, '').strip()
            if len(building) > 2:
                newWords = spell(building)
                for word in newWords:
                    words.append(pref + word)
                    words.append(word)
            return words
        else:
            return spell(building)
    
    def __addressCheck(self, building):
        #address check
        address1 = self.p.match(building)
        address2 = self.adp.match(building)
        address3 = self.x.match(building)
        if(address1 or address2 or address3):
            return "address"
        else: return False
    
    def __removeStopWords(self, building):
        building = building.replace("ภายใน", "")
        building = building.replace("ภายนอก", "")
        rmList = ["ใกล้เคียง", "บริเวณ", "รวม", "รอบ", "ข้าง", "หลัง", "หน้า", "ใน", "โซน", "นอก", "@", ":", "_", ",", "-", "¹", "³", "$", "%", "~", "{", "}", "[", "]", "(", ")", "!", "*", "?"] 
        if building.startswith(tuple(rmList)):
            for rm in rmList:
                if building.startswith(rm):
                    building = building.replace(rm, '').strip()
            if len(building) > 1:
                return building
            else:
                return ""
        else:
            return building

    def __isPerson(self, building):
        return building in self.person_corpus

    def __getCategory(self, building):
        _building = str(building).strip()
        _building = self.__removeStopWords(_building)
        address = self.__addressCheck(_building)
        if(address):
            return "address"
        else:
            b_class = self.bc.getCategory2(_building)
            if(not b_class):
                for b in _building.split(' '):
                    if len(b.strip()) > 2:
                        if self.__addressCheck(b):
                            return "address"
                        elif self.__isPerson(b.strip()):
                            print("Category of", building, ">>", b, " => Person")
                            return "person"
                        else:
                            sp = self.__correctBuilding(b)
                            for s in sp:
                                b_class = self.bc.getCategory2(s)
                                if(b_class):
                                    if b_class["type"] != "person":
                                        print("Category of", building, ">>", s, " => ", b_class["type"])
                                        break
                                    else: b_class == False
            if(b_class):
                return b_class["type"]
            else:
                return "other"
    
    def __getClassWithPlacename(self, place_namt):
        c_place = self.__getCategory(place_namt)
        if(c_place):
            return c_place
        else:
            return "other"

    def __countBuilding(self, places):
        place_namt = places.place_namt.unique()[0]
        place_id = places.place_id.unique()[0]
        place_type = places.place_type_name.unique()[0]
        place_name_class = self.__getClassWithPlacename(place_namt)
        counter = Counter(place_name_class, place_id, place_namt, place_type)
        for building in places.building_name:
            #word correction
            b_class = self.__getCategory(building)
            if(b_class):
                weight = self.bw.isBuildingWeight(building)
                self.bCount.count(counter, b_class, weight)
                if(b_class == "other"):
                    print("place_name_class:", place_name_class, " -- ", building, " => Other")
                    counter.addOther(building)
        return counter.summary()

    def transform(self, data):
        data_out = []
        place_list = data.place_id.unique()
        for place_id in place_list:
            places = data[data.place_id == place_id]
            if len(places) > 0:
        #         print(countBuilding(places))
                data_out.append(self.__countBuilding(places))
        
        out = pd.DataFrame(data_out, columns = ["place_id", "place_namt", "keyword", "address", "person", "otherworkspace", "cospace", "government", "habitat", "education", "religious", "health", "hotel", "factory", "other", "classification"])
        out.classification = out.classification.map({'สถานศึกษา':'education', 'โรงงาน/สถานที่ทำงาน':'factory', 'หมู่บ้าน/ชุมชน':'habitat', 'สาธารณสุข':'health', 'โรงแรม/รีสอร์ท':'hotel', 'ศาสนสถาน':'religious'})
        # out.to_csv("./sources/placetype_testing_data.csv", sep=',', encoding='utf-8', index=False)
        return out





