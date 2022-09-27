#!/usr/bin/python
class Counter:

    def __init__(self, keyw, cid, place_type, _class):
        self.cid = cid
        self.place_type = place_type
        self.total = 0
        self.keyword = keyw
        self.address = 0
        self.person = 0
        self.habitat = 0
        self.education = 0
        self.health = 0
        self.religious = 0
        self.hotel = 0
        self.factory = 0
        self.otherworkspace = 0
        self.cospace = 0
        self.government = 0
        self.other = []
        self._class = _class

    def addAddress(self):
        self.address += 1
        self.total += 1

    def addPerson(self):
        self.person += 1
        self.total += 1
    
    def addHabitat(self):
        self.habitat += 1
        self.total += 1

    def addEducation(self, weight):
        if weight: self.education += 5
        else: self.education += 1
        self.total += 1
    
    def addHealth(self, weight):
        if weight: self.health += 5
        else: self.health += 1
        self.total += 1

    def addReligious(self, weight):
        if weight: self.religious += 5
        else: self.religious += 1
        self.total += 1
    
    def addHotel(self, weight):
        if weight: self.hotel += 5
        else: self.hotel += 1
        self.total += 1

    def addFactory(self, weight):
        if weight: self.factory += 5
        else: self.factory += 1
        self.total += 1
    
    def addOtherWorkSpace(self):
        self.otherworkspace += 1
        self.total += 1
        
    def addCoSpace(self):
        self.cospace += 1
        self.total += 1
    
    def addGovernment(self):
        self.government += 1
        self.total += 1

    def addOther(self, item):
        self.other.append(item)
        self.total += 1
    
    def summary(self):
        return [self.cid, self.place_type, self.keyword, self.address, self.person, self.otherworkspace, self.cospace, self.government, self.habitat, self.education, self.religious, self.health, self.hotel, self.factory, len(self.other), self._class]

