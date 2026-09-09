Hide = False
Hide_Calc = False
Hide_Prod = True
Hide_Ref = True
Hide_RefNameless = False
Hide = True
DoAll= False
# DoAll= True

d=App.getDocument('EDI_with_parts')
# d=App.getDocument('EDI_Titles')
for s in d.Objects:
	if s.TypeId == 'Sketcher::SketchObject' :
		if s.Label == ActiveSketch.Label or DoAll :
			cnt=len(s.Constraints)
			for i, c in enumerate(s.Constraints):
#				print(i,'/',cnt)
				match c.Type :
					case "Coincident" :
						s.setVirtualSpace(i, Hide) 
					case "Equal" :
						s.setVirtualSpace(i, Hide) 
					case "Parallel" :
						s.setVirtualSpace(i, Hide) 
					case "Perpendicular" :
						s.setVirtualSpace(i, Hide) 
					case "Horizontal" :
						s.setVirtualSpace(i, Hide) 
					case "Tangent" :
						s.setVirtualSpace(i, Hide) 
					case "Vertical" :
						s.setVirtualSpace(i, Hide) 
					case "Coincident" :
						s.setVirtualSpace(i, Hide) 
					case "PointOnObject" :
						s.setVirtualSpace(i, Hide) 
					case "Symmetric" :
						s.setVirtualSpace(i, Hide) 
					case "Angle" | "Distance" | "DistanceY" | "DistanceX" | "Diameter" | "Radius":
						if (c.Driving == 0) or (c.Name != ''):
							s.setVirtualSpace(i, Hide_RefNameless )
							print( i,'/',cnt,"Shownx",c.Content)
						if 'Calc_' in c.Name :
							s.setVirtualSpace(i, Hide_Calc )
							print( i,'/',cnt,"Hidden Calc",c.Content)
						elif  'Prod_' in c.Name :
							s.setVirtualSpace(i, Hide_Prod)
							print( i,'/',cnt,"PROD Hidden",c.Content)
						elif  'REF_' in c.Name :
							s.setVirtualSpace(i, Hide_Ref )
#							print( i,'/',cnt,"Hidden",c.Content)
						else :
							flg=False
							exp=""
							for sub in s.ExpressionEngine:
								if sub[0] == "Constraints["+str(i)+"]":
									exp=sub[1]
									flg=Hide
									break
							s.setVirtualSpace(i, flg) 
#							print("Setv", i,'/',cnt,flg,exp,c.Content)

									#
					case _ :
						print(i,c.Type)
print("Done")
