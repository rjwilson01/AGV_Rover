d=App.getDocument('EDI_with_parts')
d.FileName
import os
(Pth, FileName)=os.path.split(d.FileName)
Pth=Pth+"/exports/"
head=Pth+FileName
from pathlib import Path
DirPath=Path(Pth)

DirPath.mkdir(parents=True, exist_ok=True)


myfilecon=open(head+".con.csv", "w")
myfilegeo=open(head+".geo.csv", "w")

print('Object,Index,Label,Value,Type,Active,Driving,Name,Content',file=myfilecon)
print('Object,Index,Label,Content',file=myfilegeo)
for s in d.Objects:
	if s.TypeId == 'Sketcher::SketchObject' :
		print(s.Label)
		i=0
		for c in s.Constraints:
			print('Constraint,',i,',',s.Label,',',c.Value,',',c.Type,',',c.IsActive,',',c.Driving,',',c.Name,',"',c.Content.replace('"','""').replace('\n','~'),'"',file=myfilecon)
#			print(i,',',s.Label,',',c.Value,',',c.Type,',',c.IsActive,',',c.Driving,',',c.Name,",'",c.Content.replace('"','""').replace('\n','~'),"'")
			i=i+1
		i=0
		for g in s.Geometry:
			print('Geometry,',i,',',s.Label,",'",g.Content.replace('\n','~'),"'",file=myfilegeo)
			i=i+1

myfilecon.close()
myfilegeo.close()
myfile=open(head+".dep.txt", "w")
print(App.ActiveDocument.DependencyGraph,file=myfile)
myfile.close()            

myfile=open(head+".base.csv", "w")
print('Label,Name,Angle,x,y,z',file=myfile)
for s in d.Objects:
	if s.TypeId == 'Sketcher::SketchObject' :
		print(s.Label,',',s.Name,',',s.Placement.Rotation.Angle,',',s.Placement.Base.x,',',s.Placement.Base.y,',',s.Placement.Base.z,file=myfile)
		print(s.Label,',',s.Name,',',s.Placement.Rotation.Angle,',',s.Placement.Base.x,',',s.Placement.Base.y,',',s.Placement.Base.z,s.TypeId) 
myfile.close()            
print('DONE') 