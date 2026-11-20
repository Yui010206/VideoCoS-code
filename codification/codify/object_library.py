"""Procedural, editable semantic stand-ins fitted to bundle local bounds.

Promoted from the archived ADT builder (code/codification/stages/object_library.py,
sha256 a3328edeefa2b4d60fe6170ef32833211776ab77d0b0ec3c96de4f9744936c62); the
recipes are unchanged. Changes: canonical up/front axes are required instead of
ADT defaults, provenance names are neutral (bounds may be reference or estimated),
and environment shells are not built here.

API: build_object(instance, [xmin,xmax,ymin,ymax,zmin,zmax], color=(r,g,b,a))
returns a single bpy.types.Object in local space; the caller applies the pose.
"""
import math
import hashlib
import bpy
from mathutils import Vector, Matrix

TAU = 2.0 * math.pi
_MATERIALS = {}


def _material(name, color, metallic=0.0, roughness=0.55):
    # Public color input and the artist-authored palette are sRGB; Blender's
    # shader sockets store linear scene-referred RGB.
    rgba = tuple(float(x)/12.92 if float(x)<=.04045 else ((float(x)+.055)/1.055)**2.4
                 for x in color[:3]) + (color[3] if len(color) > 3 else 1.0,)
    key = (name, tuple(round(float(x), 4) for x in rgba), metallic, roughness)
    if key in _MATERIALS and _MATERIALS[key].name in bpy.data.materials:
        return _MATERIALS[key]
    mat = bpy.data.materials.new('proxy_' + name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    p = mat.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = rgba
    p.inputs['Metallic'].default_value = metallic
    p.inputs['Roughness'].default_value = roughness
    _MATERIALS[key] = mat
    return mat


class _Mesh:
    """One mesh assembled directly; coordinates use Z up and front = -Y."""
    def __init__(self, materials):
        self.vertices, self.faces, self.face_materials, self.smooth = [], [], [], []
        self.materials = materials

    def part(self, verts, faces, mat=0, smooth=False):
        start = len(self.vertices)
        self.vertices.extend(verts)
        self.faces.extend(tuple(start + v for v in f) for f in faces)
        self.face_materials.extend([mat] * len(faces))
        self.smooth.extend([smooth] * len(faces))

    def box(self, p, d, mat=0):
        x,y,z = p; a,b,c = [v * .5 for v in d]
        verts = [(x+sx*a,y+sy*b,z+sz*c) for sx,sy,sz in
                 [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                  (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        self.part(verts,[(0,3,2,1),(4,5,6,7),(0,1,5,4),
                         (1,2,6,5),(2,3,7,6),(3,0,4,7)],mat)

    def lathe(self, p, profile, radius=(1,1), mat=0, n=28, caps=True):
        # profile = ordered (height, radius) pairs, including inner surface
        x,y,z = p; rx,ry = radius
        vs=[(x+rx*r*math.cos(TAU*j/n),y+ry*r*math.sin(TAU*j/n),z+h)
            for h,r in profile for j in range(n)]
        fs=[]
        for i in range(len(profile)-1):
            fs += [(i*n+j,i*n+(j+1)%n,(i+1)*n+(j+1)%n,(i+1)*n+j)
                   for j in range(n)]
        if caps:
            fs += [tuple(reversed(range(n))),tuple((len(profile)-1)*n+j for j in range(n))]
        self.part(vs,fs,mat,True)

    def cylinder(self, p, r, h, mat=0, n=24, rtop=None):
        self.lathe(p,[(-h*.5,r),(h*.5,r if rtop is None else rtop)],mat=mat,n=n)

    def ellipsoid(self, p, d, mat=0, n=20, rings=10):
        x,y,z=p; a,b,c=[v*.5 for v in d]
        vs=[(x+a*math.sin(math.pi*i/rings)*math.cos(TAU*j/n),
             y+b*math.sin(math.pi*i/rings)*math.sin(TAU*j/n),
             z+c*math.cos(math.pi*i/rings)) for i in range(rings+1) for j in range(n)]
        fs=[(i*n+j,(i+1)*n+j,(i+1)*n+(j+1)%n,i*n+(j+1)%n)
            for i in range(rings) for j in range(n)]
        self.part(vs,fs,mat,True)

    def rod(self, a, b, radius=.02, mat=0, n=10):
        a,b=Vector(a),Vector(b); direction=b-a
        if direction.length < 1e-8: return
        direction.normalize()
        u=direction.cross(Vector((0,0,1)))
        if u.length < .001: u=direction.cross(Vector((0,1,0)))
        u.normalize(); v=direction.cross(u)
        vs=[tuple(p+radius*(u*math.cos(TAU*j/n)+v*math.sin(TAU*j/n)))
            for p in (a,b) for j in range(n)]
        fs=[(j,(j+1)%n,n+(j+1)%n,n+j) for j in range(n)]
        fs.extend([tuple(reversed(range(n))),tuple(n+j for j in range(n))])
        self.part(vs,fs,mat,True)

    def tube(self, points, radius=.025, mat=0, n=8):
        for a,b in zip(points[:-1],points[1:]): self.rod(a,b,radius,mat,n)

    def loop(self, center, radii, plane='XZ', mat=0, thickness=.025, steps=28):
        points=[]
        for i in range(steps+1):
            p=list(center); p['XYZ'.index(plane[0])]+=radii[0]*math.cos(TAU*i/steps)
            p['XYZ'.index(plane[1])]+=radii[1]*math.sin(TAU*i/steps);points.append(p)
        self.tube(points,thickness,mat)

    def rotate_z(self):
        self.vertices=[(-y,x,z) for x,y,z in self.vertices]

    def align_thin_axis(self, source_axis, target_axis):
        """Rotate a flat recipe so its face normal follows the thin axis of the bounds.

        These are proper rotations, not axis swaps/reflections. This must run
        before the canonical-to-local orientation and the exact-bounds fit.
        """
        if source_axis==target_axis:return
        axes={source_axis,target_axis}
        if axes=={0,1}:self.vertices=[(-y,x,z) for x,y,z in self.vertices]
        elif axes=={1,2}:self.vertices=[(x,-z,y) for x,y,z in self.vertices]
        else:self.vertices=[(z,y,-x) for x,y,z in self.vertices]


def _base_color(instance, color):
    name=instance.get('instance_name','').lower()
    # Explicit object names are useful even when sampled pixels include shadows.
    for token,c in [('black',(.035,.04,.045,1)),('white',(.82,.80,.74,1)),
                    ('yellow',(.83,.49,.07,1)),('grey',(.31,.33,.34,1)),
                    ('red',(.62,.075,.045,1)),('wooden',(.40,.22,.10,1)),
                    ('bamboo',(.63,.43,.23,1))]:
        if token in name: return c
    return tuple(color[:3])+(1.0,)


def _palette(instance, color):
    base=_base_color(instance,color)
    return [_material('surface',base),
            _material('dark',(.022,.026,.032,1),.12,.45),
            _material('wood',(.32,.17,.075,1),0,.62),
            _material('metal',(.43,.47,.49,1),.8,.27),
            _material('cream',(.86,.83,.74,1)),
            _material('glass_opaque',(.15,.23,.24,1),.25,.18),
            _material('foliage',(.055,.20,.038,1),0,.85),
            _material('accent',(.66,.24,.045,1)),
            _material('paper',(.76,.72,.61,1),0,.9),
            _material('screen',(.008,.012,.018,1),.3,.18)]


def _legs(m, top=.42, bottom=-.5, radius=.035, mat=2, spread=.42, depth=.39):
    for x in (-spread,spread):
        for y in (-depth,depth): m.rod((x,y,bottom),(x*.88,y*.88,top),radius,mat)


def _hollow_box(m, base=-.5, top=.5, thickness=.045, mat=0, back=True):
    m.box((0,0,base+thickness*.5),(1,1,thickness),mat)
    for x in (-.5+thickness*.5,.5-thickness*.5):
        m.box((x,0,(base+top)*.5),(thickness,1,top-base),mat)
    if back: m.box((0,.5-thickness*.5,(base+top)*.5),(1,thickness,top-base),mat)


def _handle(m, p=(.3,-.53,0), length=.28, mat=3, vertical=True):
    x,y,z=p
    if vertical:
        m.rod((x,y,z-length*.5),(x,y,z+length*.5),.015,mat)
        for h in (-length*.5,length*.5):m.rod((x,y,z+h),(x,y+.06,z+h),.012,mat)
    else:
        m.rod((x-length*.5,y,z),(x+length*.5,y,z),.015,mat)
        for h in (-length*.5,length*.5):m.rod((x+h,y,z),(x+h,y+.06,z),.012,mat)


def _perforated_plate(m, width=.86, depth=.86, z=.46, thickness=.08,
                      cols=7, rows=5, rx=.036, ry=.036, mat=4):
    """A rectangular slab with real through-holes, generated without booleans."""
    hx=width/(2*cols);hy=depth/(2*rows)
    angles=sorted(set([TAU*k/16 for k in range(16)]+
                      [math.atan2(sy*hy,sx*hx)%TAU for sx in (-1,1) for sy in (-1,1)]))
    segments=len(angles)
    for col in range(cols):
        for row in range(rows):
            cx=-width*.5+(col+.5)*width/cols
            cy=-depth*.5+(row+.5)*depth/rows
            vs=[]
            for height in (z-thickness*.5,z+thickness*.5):
                for inner in (False,True):
                    for a in angles:
                        c,s=math.cos(a),math.sin(a)
                        r=1/max(abs(c)/hx,abs(s)/hy)
                        x,y=(rx*c,ry*s) if inner else (r*c,r*s)
                        vs.append((cx+x,cy+y,height))
            fs=[]
            for k in range(segments):
                j=(k+1)%segments
                fs.append((k,segments+k,segments+j,j))
                fs.append((2*segments+k,2*segments+j,3*segments+j,3*segments+k))
                fs.append((segments+k,3*segments+k,3*segments+j,segments+j))
            m.part(vs,fs,mat)
    for y in (-depth*.5,depth*.5):m.box((0,y,z),(width,.012,thickness),mat)
    for x in (-width*.5,width*.5):m.box((x,0,z),(.012,depth,thickness),mat)


def _recipe(m, category, name, dims):
    """Create unit-sized semantic geometry; final exact-bounds fit is separate."""
    c=category.lower(); n=name.lower(); W,D,H=dims
    # A few dataset categories describe the wrong physical object. Preserve
    # original category metadata but choose geometry using the instance name.
    if 'displayunit' in n:
        result=_recipe(m,'bookcase',name.replace('DisplayUnit','Bookcase'),dims)
        if W<D:m.rotate_z()
        return 'display_unit_shelving_name_override'
    if 'bookbox' in n:c='book'
    if c in ('dining table','coffee table','console table','dressing table','table'):
        island='island' in n
        if island:
            # Reference footage shows a butcher-block top on an open black
            # frame, with a low shelf; it is not a fitted cabinet island.
            wood=len(m.materials)
            m.materials.append(_material('island_butcher_block',(.48,.31,.17,1),0,.57))
            m.box((0,0,.445),(1,1,.11),wood)
            for x in (-.43,.43):
                for y in (-.40,.40):m.box((x,y,-.055),(.065,.075,.89),1)
            for y in (-.40,.40):m.box((0,y,.35),(.88,.065,.095),1)
            for x in (-.43,.43):m.box((x,0,.35),(.065,.85,.095),1)
            m.box((0,0,-.30),(.88,.84,.045),1)
            for y in (-.40,.40):m.box((0,y,-.33),(.88,.05,.065),1)
            return 'butcher_block_island_open_black_frame'
        elif 'round' in n:
            m.cylinder((0,0,.45),.5,.1,0,n=48);_legs(m,.4,-.5,.035)
        else:
            m.box((0,0,.45),(1,1,.10),0);_legs(m,.4,-.5,.035)
        if c=='dressing table':
            m.box((0,0,.28),(.93,.9,.23),0)
            for x in (-.25,.25):_handle(m,(x,-.48,.27),.16,3,False)
        return 'table_with_legs'
    if c in ('chair','dining chair','bar stool','armchair','couch'):
        padded=c in ('armchair','couch') or 'wing' in n
        if padded:
            _legs(m,-.27,-.5,.045,2,.39,.36)
            m.box((0,0,-.12),(.93,.89,.27),0)
            count=3 if c=='couch' else 1
            for k in range(count):
                x=(k-(count-1)/2)*(.79/count)
                m.ellipsoid((x,-.08,.025),(.78/count,.74,.22),0)
                m.ellipsoid((x,.34,.30),(.79/count,.26,.43),0)
            for x in (-.44,.44):m.ellipsoid((x,0,.14),(.18,.92,.43),0)
        else:
            seat_z=.10 if c=='bar stool' else -.06
            _legs(m,seat_z-.045,-.5,.027,1)
            m.box((0,0,seat_z),(.89,.85,.09),0)
            for x in (-.38,.38):m.rod((x,.35,seat_z),(x,.39,.5),.027,1)
            m.box((0,.39,.37),(.89,.085,.25),0)
            if c=='bar stool':
                for y in (-.34,.34):m.rod((-.39,y,-.20),(.39,y,-.20),.025,1)
        return 'upholstered_seating' if padded else 'chair_with_back_and_legs'
    if c in ('cabinet','bookcase','cabinets and shelves','tv stand','side table','shoe rack','wine rack'):
        if 'utilitycart' in n:
            for x in (-.45,.45):
                for y in (-.43,.43):
                    m.rod((x,y,-.39),(x,y,.50),.022,1)
                    m.ellipsoid((x,y,-.445),(.075,.12,.11),1,n=12,rings=6)
                    m.rod((x,y,-.43),(x,y,-.34),.018,3)
            for z in (-.33,.04,.42):
                m.box((0,0,z),(.95,.92,.035),1)
                for y in (-.45,.45):m.box((0,y,z+.045),(.95,.025,.09),1)
                for x in (-.465,.465):m.box((x,0,z+.045),(.025,.92,.09),1)
            return 'open_three_tier_cart_with_casters'
        is_shelf=c in ('bookcase','wine rack','shoe rack') or any(x in n for x in ('shel','cart'))
        _hollow_box(m,-.43,.48,.045,0,back=not ('wire' in n or 'cart' in n))
        m.box((0,0,.48),(1,1,.04),0)
        _legs(m,-.40,-.5,.03,1,.44,.40)
        levels=5 if H>1.4 else 3
        if is_shelf:
            for k in range(1,levels):m.box((0,0,-.43+.91*k/levels),(.96,.98,.035),0)
            if c=='wine rack':
                for x in (-.25,0,.25):m.box((x,0,0),(.025,.98,.9),0)
        else:
            doors=2 if W>.55 else 1
            for k in range(doors):
                x=(k-(doors-1)*.5)/doors
                m.box((x,-.505,.02),(.94/doors,.035,.88),5 if 'glass' in n else 0)
                _handle(m,(x+.30/doors,-.545,.08),.2,3)
        return 'open_shelves' if is_shelf else 'cabinet_with_doors'
    if c=='shelf' or c=='rack' or c=='spice rack' or c=='dishrack':
        if 'wire' in n:
            for z in (-.48,-.16,.16,.48):
                for x in (-.47,.47):m.rod((x,-.45,z),(x,.45,z),.015,1)
                for y in [-.45+i*.10 for i in range(10)]:m.rod((-.47,y,z),(.47,y,z),.008,1)
            _legs(m,.5,-.5,.018,1,.47,.45)
        elif 'vinyl' in n or 'ledge' in n:
            m.box((0,0,-.43),(1,1,.14),0);m.box((0,.43,0),(1,.14,1),0)
            m.box((0,-.43,-.26),(1,.14,.45),0)
        elif c in ('spice rack','dishrack','rack'):
            for z in (-.44,0,.44):m.box((0,0,z),(1,.95,.045),0)
            _legs(m,.5,-.5,.022,1,.46,.43)
        else:
            m.box((0,0,.38),(1,1,.24),0)
            for x in (-.35,.35):
                m.rod((x,.42,.34),(x,.42,-.5),.025,3)
                m.rod((x,-.35,.25),(x,.42,-.5),.025,3)
        return 'shelf_with_supports'
    if c=='part of a cabinet/wardrobe' or c=='door':
        if 'drawer' in n:
            _hollow_box(m,-.5,.45,.06,2)
            m.box((0,-.48,0),(1,.08,1),0);_handle(m,(0,-.55,.23),.34,3,False)
        else:
            m.box((0,0,0),(1,.15,1),0)
            pane_material=9 if 'doubledoor' in n else 4
            for z in (-.25,.25):m.box((0,-.09,z),(.78,.03,.36),pane_material)
            _handle(m,(.38,-.16,-.04),.09 if c=='door' else .22,3,vertical=c!='door')
            m.align_thin_axis(1,min(range(3),key=lambda k:dims[k]))
        return 'panel_with_handle'
    if c=='refrigerator':
        m.box((0,.04,0),(1,.92,1),0)
        m.box((0,-.48,.30),(.98,.08,.38),0);m.box((0,-.48,-.20),(.98,.08,.59),0)
        _handle(m,(.36,-.56,.25),.24,3);_handle(m,(.36,-.56,-.1),.37,3)
        m.box((0,-.495,.10),(.98,.011,.014),1)
        return 'refrigerator_doors'
    if c in ('television','smart home video','laptop computer','consumer electronic device','router','computer keyboard','remote control','electronic device control'):
        if c in ('television','smart home video','laptop computer'):
            m.box((0,.05,.08),(1,.13,.84),1);m.box((0,-.025,.08),(.93,.02,.76),9)
            m.rod((0,.08,-.5),(0,.08,-.27),.045,1)
            m.box((0,0,-.48),(.54,.46,.04),1)
        elif c=='computer keyboard':
            m.box((0,0,-.2),(1,1,.6),1)
            for row in range(5):
                for col in range(14):m.box((-.46+col*.071,-.39+row*.19,.24),(.059,.15,.25),0)
        else:
            m.box((0,0,0),(1,1,.8),0)
            for k in range(3):m.ellipsoid((-.22+.22*k,-.51,.05),(.04,.02,.04),3)
            if c=='router':
                for x in (-.4,.4):m.rod((x,.28,.15),(x,.28,.7),.023,1)
            if c in ('remote control','electronic device control'):
                for x in (-.22,.22):
                    for y in (-.3,-.1,.1,.3):m.ellipsoid((x,y,.43),(.17,.10,.09),1)
        return 'electronic_device'
    if c in ('cup','bowl','pet bowl','plate','pot','frying pan','tray','baking pan','tea kettle','vase','candle','can','jar','container','bottle'):
        if c in ('cup','bowl','pet bowl','pot','frying pan'):
            radius=.36 if c=='cup' else .5
            profile=[(-.5,.75),(-.43,.87),(.45,1),(.5,1),(.5,.90),(-.35,.69)]
            m.lathe((0,0,0),profile,(radius,radius),0)
            if c=='cup':m.loop((.43,0,.05),(.20,.31),'XZ',0,.055)
            if c=='pot':
                for x in (-.54,.54):m.loop((x,0,.25),(.13,.16),'XY',1,.04)
            if c=='frying pan':m.box((.74,0,.10),(.65,.19,.18),1)
        elif c in ('plate','tray','baking pan'):
            if c=='plate' or 'candle' in n:
                m.lathe((0,0,0),[(-.5,.30),(-.43,.40),(.15,.50),(.5,.49),(.43,.44),(-.15,.34)],mat=0,n=40)
            else:
                _hollow_box(m,-.5,.5,.08,0);m.box((0,-.46,0),(1,.08,1),0)
                if c=='baking pan':
                    for x in (-.29,0,.29):
                        for y in (-.27,.27):m.lathe((x,y,.15),[(0,.11),(.25,.13),(.25,.10),(0,.08)],mat=1,n=16)
        elif c=='tea kettle':
            m.lathe((0,0,-.1),[(-.4,.27),(-.25,.41),(.10,.42),(.3,.27)],mat=0)
            m.cylinder((0,0,.24),.28,.05,3);m.ellipsoid((0,0,.31),(.13,.13,.12),1)
            m.tube([(.33,0,.17),(.57,0,.35),(.52,0,-.22),(.32,0,-.30)],.065,1)
            m.rod((-.28,0,-.1),(-.54,0,.13),.095,3)
        elif c=='vase':
            m.lathe((0,0,0),[(-.5,.22),(-.45,.31),(-.1,.5),(.2,.36),(.45,.20),(.5,.21),(.5,.17),(.1,.16)],mat=0)
        elif c=='candle':
            m.cylinder((0,0,-.03),.48,.94,0);m.cylinder((0,0,.45),.43,.025,4)
            m.rod((0,0,.46),(0,0,.53),.012,1)
        elif c=='bottle':
            m.lathe((0,0,0),[(-.5,.35),(.10,.36),(.27,.17),(.45,.16)],mat=0)
            m.cylinder((0,0,.47),.18,.08,1)
        elif c=='container' and ('box' in n or 'whitecontainer' in n):
            _hollow_box(m,-.5,.45,.05,0);m.box((0,-.475,0),(1,.05,.95),0);m.box((0,0,.47),(1,1,.06),4)
        elif c in ('container','jar'):
            # an open vessel things can be put into (outer wall, rim, inner wall, floor); a lid is its own object
            m.lathe((0,0,0),[(-.5,.96),(.5,1.0),(.5,.93),(-.44,.89)],(.5,.5),0)
        else:
            m.cylinder((0,0,-.03),.48,.94,0);m.cylinder((0,0,.46),.50,.08,3)
            m.cylinder((0,0,.02),.486,.45,4)
        return 'lathed_household_object'
    if c in ('house plant',):
        m.lathe((0,0,-.5),[(0,.21),(.31,.29),(.34,.30),(.34,.255),(.26,.24)],mat=0)
        m.cylinder((0,0,-.19),.25,.025,2)
        for i in range(11):
            a=i*2.399; z=-.1+(i%4)*.15; r=.3 if i%2 else .42
            endpoint=(math.cos(a)*r,math.sin(a)*r,z+.18)
            m.rod((0,0,-.2),endpoint,.009,6)
            m.ellipsoid(endpoint,(.24,.18,.32),6,n=12,rings=6)
        m.ellipsoid((0,0,.39),(.19,.17,.24),6)
        return 'potted_foliage'
    if c in ('clothes basket','trash can','box','decorative accessory','tissue box'):
        if c=='clothes basket':
            m.box((0,0,-.48),(1,1,.04),0)
            for z in [-.45+i*.1 for i in range(10)]:
                for y in (-.47,.47):m.rod((-.47,y,z),(.47,y,z),.013,0)
                for x in (-.47,.47):m.rod((x,-.47,z),(x,.47,z),.013,0)
            for i in range(8):
                t=-.47+i*.134
                for y in (-.47,.47):m.rod((t,y,-.48),(t,y,.48),.014,0)
                for x in (-.47,.47):m.rod((x,t,-.48),(x,t,.48),.014,0)
        elif c=='trash can':
            m.lathe((0,0,0),[(-.5,.36),(.43,.49),(.48,.5),(.48,.46),(-.42,.33)],mat=0)
            if 'lid' in n:m.ellipsoid((0,0,.45),(1,1,.1),0)
        elif c=='tissue box':
            m.box((0,0,-.12),(1,1,.76),0);m.box((0,0,.27),(.40,.16,.025),1)
            m.part([(-.18,0,.29),(.15,0,.29),(.20,.05,.5),(-.12,-.05,.46)],[(0,1,2,3)],4)
        elif 'hex' in n:
            m.cylinder((0,0,0),.5,1,0,n=6);m.cylinder((0,0,.46),.51,.08,4,n=6)
        elif 'decoration' in n:
            m.lathe((0,0,0),[(-.5,.26),(-.20,.5),(.10,.31),(.5,.1)],mat=0,n=8)
        else:
            m.box((0,0,0),(1,1,1),0);m.box((0,0,.47),(1.01,1.01,.06),4)
        return 'container_or_decoration'
    if c in ('book','notebook','magazine') or 'bookbox' in n:
        thin_axis=min(range(3),key=lambda k:dims[k])
        # Thin dimension can be any axis because ground-truth poses are not a
        # canonicalized CAD asset library; fit to actual local extents below.
        m.box((0,0,0),(.94,.88,.90),8)
        m.box((-.485,0,0),(.03,1,1),0);m.box((.485,0,0),(.03,1,1),0)
        m.box((0,.47,0),(1,.06,1),0)
        for z in (-.35,.25):m.box((0,-.46,z),(.94,.015,.017),4)
        m.align_thin_axis(0,thin_axis)
        return 'book_with_pages_and_cover'
    if c in ('wall artwork','picture','wall clock','desk clock'):
        if 'clock' in c:
            m.cylinder((0,0,0),.5,.12,0,n=48)
            m.cylinder((0,0,.07),.44,.015,4,n=48)
            m.rod((0,0,.09),(.02,.30,.09),.016,1)
            m.rod((0,0,.09),(-.19,-.08,.09),.02,1)
            m.vertices=[(x,-z,y) for x,y,z in m.vertices]
        else:
            m.box((0,0,0),(1,.08,1),1)
            m.box((0,-.047,0),(.87,.02,.87),4)
            seed=int(hashlib.md5(n.encode()).hexdigest()[:8],16)
            for i in range(5):
                x=((seed>>(i*3))%8)/14-.25;z=((seed>>(i*4))%8)/14-.25
                m.ellipsoid((x,-.065,z),(.18,.015,.25),7 if i%2 else 0,n=10,rings=6)
        m.align_thin_axis(1,min(range(3),key=lambda k:dims[k]))
        return 'framed_wall_object'
    if c in ('food object',):
        if 'banana' in n:
            points=[(-.44+i*.11,0,-.20+.55*((i/8-.5)*2)**2) for i in range(9)]
            m.tube(points,.11,7,n=12)
        elif 'carrot' in n:
            m.lathe((0,0,0),[(-.5,.015),(.35,.18),(.46,.13)],mat=7)
            for x in (-.08,0,.08):m.rod((0,0,.4),(x,.02,.58),.02,6)
        elif 'donut' in n:
            m.loop((0,0,0),(.32,.32),'XY',2,.17,40)
        elif 'sandwich' in n or 'cake' in n:
            for z,mat in ((-.36,2),(-.14,6),(.0,7),(.15,4),(.36,2)):
                m.box((0,0,z),(1,1,.22),mat)
        elif 'spice' in n:
            m.cylinder((0,0,-.06),.40,.88,0);m.cylinder((0,0,.43),.43,.14,1)
        else:
            m.ellipsoid((0,0,-.03),(1,.96,.91),0)
            m.rod((0,0,.37),(.04,0,.54),.025,2)
            if 'pear' in n:m.ellipsoid((0,0,.20),(.56,.55,.56),0)
        return 'food_approximation'
    if c in ('coffee maker','blender','food processor','toaster','cooktop'):
        if c=='coffee maker':
            m.box((0,.22,0),(1,.54,1),0);m.box((0,-.14,.38),(1,.65,.24),0)
            m.box((0,-.15,-.46),(1,.7,.08),1)
            m.lathe((0,-.22,-.37),[(0,.20),(.18,.27),(.45,.22)],mat=5)
            m.loop((.27,-.22,-.12),(.13,.17),'XZ',1,.035)
        elif c in ('blender','food processor'):
            m.lathe((0,0,-.5),[(0,.42),(.30,.40),(.35,.31)],mat=0)
            m.lathe((0,0,-.15),[(0,.27),(.55,.39)],mat=5)
            m.cylinder((0,0,.44),.41,.11,1)
            m.loop((.41,0,.13),(.16,.22),'XZ',1,.04)
        elif c=='toaster':
            m.box((0,0,0),(1,1,.93),0)
            for x in (-.24,.24):m.box((x,0,.49),(.13,.74,.03),1)
            m.box((.51,0,.12),(.04,.24,.1),1)
        else:
            m.box((0,0,0),(1,1,.75),1)
            for x in (-.23,.23):
                for y in (-.23,.23):m.loop((x,y,.4),(.17,.17),'XY',3,.015)
        return 'kitchen_appliance'
    if c in ('air purifier','dehumidifier','speaker','smart home speaker'):
        if 'speaker' in c:
            m.box((0,0,0),(1,.84,1),0)
            for z in (-.25,.20):
                before=len(m.vertices);m.cylinder((0,0,0),.29,.025,1,n=28)
                m.vertices[before:]=[(x,-.44-h,z+y) for x,y,h in m.vertices[before:]]
        else:
            m.box((0,0,0),(1,1,1),0)
            for z in [-.33+i*.07 for i in range(10)]:m.box((0,-.505,z),(.78,.015,.018),1)
            m.cylinder((0,0,.51),.08,.01,1)
        return 'vented_appliance'
    if c in ('desk lamp','light'):
        if c=='desk lamp':
            m.cylinder((0,0,-.46),.41,.08,0)
            m.rod((0,0,-.42),(0,0,.19),.035,3)
            m.lathe((0,0,.18),[(0,.5),(.32,.26)],mat=0)
        else:
            m.ellipsoid((0,0,0),(1,.75,.8),0)
            m.ellipsoid((-.2,0,.28),(.4,.6,.4),0)
            m.ellipsoid((.2,0,.28),(.4,.6,.4),0)
        return 'lamp'
    if c in ('bed frame','mattress','pillow','napkin','mouse pad','curtain'):
        if c=='bed frame':
            m.box((0,0,-.2),(1,1,.20),0);m.box((0,.46,.13),(1,.08,.74),0)
            _legs(m,-.25,-.5,.035,1)
            for y in [-.42+i*.12 for i in range(8)]:m.box((0,y,-.08),(.92,.045,.03),2)
        elif c in ('mattress','pillow'):
            m.ellipsoid((0,0,0),(1,1,1),0,n=24,rings=12)
        elif c=='curtain':
            for i in range(15):
                x=-.47+i*.067
                m.ellipsoid((x,math.sin(i*1.6)*.30,0),(.1,.35,1),0,n=10,rings=6)
        else:m.box((0,0,0),(1,1,1),0)
        return 'soft_furnishing'
    if c=='step stool':
        # The carried StepStool has a white plastic shell and a distinctive
        # perforated top (reference frame 001923), not a metal two-step ladder.
        white=len(m.materials)
        m.materials.append(_material('white_stool_plastic',(.92,.93,.94,1),0,.4))
        _perforated_plate(m,rx=.036,ry=min(.06,.036*W/max(D,1e-5)),mat=white)
        for sx in (-1,1):
            for sy in (-1,1):
                top=(sx*.335,sy*.335,.40);bottom=(sx*.425,sy*.425,-.50)
                vs=[]
                for center in (bottom,top):
                    for dx,dy in ((-.075,-.075),(.075,-.075),(.075,.075),(-.075,.075)):
                        vs.append((center[0]+dx,center[1]+dy,center[2]))
                m.part(vs,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),
                           (2,3,7,6),(3,0,4,7)],white)
        # Broad upper side skirts create the characteristic plastic shell,
        # leaving large openings under them between the four flared legs.
        for x in (-.37,.37):m.box((x,0,.24),(.075,.79,.31),white)
        for y in (-.37,.37):m.box((0,y,.24),(.79,.075,.31),white)
        return 'white_perforated_plastic_step_stool'
    if c=='step ladder':
        levels=3
        for i in range(levels):
            y=.33-i*.66/max(1,levels-1);z=-.25+i*.65/max(1,levels-1)
            m.box((0,y,z),(.92,.34,.07),0)
        for x in (-.43,.43):
            m.rod((x,-.45,-.5),(x,.35,.5),.028,3)
            m.rod((x,.48,-.5),(x,.35,.5),.028,3)
        return 'stepped_stool'
    if c in ('freestanding accessory','cleaning utensil','spray bottle','liquid soap dispenser'):
        m.lathe((0,0,-.5),[(0,.33),(.12,.44),(.6,.4),(.75,.21)],mat=0)
        m.cylinder((0,0,.29),.16,.15,1)
        if c in ('spray bottle','cleaning utensil','liquid soap dispenser'):
            m.box((.10,0,.43),(.47,.20,.14),0)
            m.rod((.12,0,.38),(.22,0,.2),.035,1)
        else:m.cylinder((0,0,.44),.19,.12,0)
        return 'bottle_with_dispenser'
    if c in ('fork','spoon','cutting board','kitchen utensil','scissors'):
        if 'jar' in n or 'canister' in n:
            m.cylinder((0,0,0),.45,.88,0);m.cylinder((0,0,.46),.5,.08,1)
        elif c=='cutting board':
            m.box((0,0,0),(1,1,.8),2);m.box((0,.48,0),(.36,.12,.8),2)
        elif c=='scissors':
            for x in (-.23,.23):m.loop((x,-.3,0),(.22,.20),'XY',0,.045)
            m.rod((-.2,-.1,0),(.19,.5,0),.04,3);m.rod((.2,-.1,0),(-.19,.5,0),.04,3)
        elif 'knife' in n:
            m.box((0,-.27,0),(.19,.46,.2),1)
            m.part([(-.08,-.06,-.025),(.19,-.06,-.025),(.12,.5,-.025),(-.08,.5,-.025),
                    (-.08,-.06,.025),(.19,-.06,.025),(.12,.5,.025),(-.08,.5,.025)],
                   [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],3)
        elif c=='spoon':
            m.rod((0,-.5,0),(0,.15,0),.04,2)
            m.ellipsoid((0,.28,0),(.38,.43,.10),2)
        elif c=='fork':
            m.rod((0,-.5,0),(0,.22,0),.04,2)
            for x in (-.12,-.04,.04,.12):m.rod((0,.12,0),(x,.5,0),.022,2)
        else:
            m.cylinder((0,0,-.46),.5,.08,3);m.rod((0,0,-.42),(0,0,.46),.04,3)
            for z in (-.2,.1,.4):m.loop((0,0,z),(.35,.35),'XY',3,.02)
        return 'kitchen_tool'
    if c in ('wall-mounted coat rack','hook','keys','accessory'):
        m.box((0,.15,0),(1,.15,.3),0)
        for x in (-.30,0,.30):m.tube([(x,.1,.1),(x,-.4,-.2),(x,-.4,.3)],.04,3)
        return 'hook_or_hanger'
    if c in ('musical instrument',):
        if 'vinylmusic' in n:
            m.cylinder((0,0,0),.5,.02,1,n=48);m.cylinder((0,0,.012),.14,.006,0,n=32)
            thin_axis=min(range(3),key=lambda k:dims[k])
            m.align_thin_axis(2,thin_axis)
        else:
            m.box((0,0,0),(1,1,.65),0);m.cylinder((-.1,0,.35),.35,.06,1,n=40)
            m.tube([(.38,.35,.4),(.35,-.1,.4),(.13,-.25,.4)],.018,3)
        return 'record_or_player'
    if c in ('figurine','toy','play set','hat','video game controller','computer mouse'):
        if c in ('computer mouse','video game controller'):
            m.ellipsoid((0,0,0),(1,1,1),0)
            if c=='video game controller':
                for x in (-.25,.25):m.cylinder((x,0,.42),.10,.15,1,n=16)
        elif c=='hat':
            m.ellipsoid((0,.15,.08),(.85,.8,.85),0);m.ellipsoid((0,-.28,-.28),(1,.75,.12),0)
        else:
            m.ellipsoid((0,0,-.08),(.64,.50,.62),0)
            m.ellipsoid((0,-.04,.32),(.45,.42,.39),0)
            for x in (-.23,.23):
                m.ellipsoid((x,-.05,-.38),(.21,.4,.25),0)
                m.rod((x*.75,0,.12),(x*1.8,-.1,-.06),.07,0)
        return 'organic_approximation'
    if c in ('thermostat','safety sign','surveillance camera','wallet','coaster','built in'):
        m.box((0,0,0),(1,1,1),0)
        if c in ('thermostat','surveillance camera'):
            m.box((0,-.51,.12),(.62,.03,.35),9)
        if c=='built in':
            for x in [-.4+i*.1 for i in range(9)]:
                for z in [-.4+i*.1 for i in range(9)]:m.ellipsoid((x,-.505,z),(.025,.01,.025),1,n=8,rings=4)
        return 'simple_semantic_accessory'
    if c in ('lid','cap','cover'):
        # a flat disc with a raised rim, fitted to the bounds: a lid that lifts off its container
        m.lathe((0,0,0),[(-.5,1.0),(.5,1.0),(.5,.9),(.1,.9)],(.5,.5),0)
        return 'lid'
    m.box((0,0,0),(1,1,1),0)
    return 'bounding_box_fallback'


def build_object(instance, bounds, color=(.5,.5,.5,1.0)):
    """Return a single local-space mesh fitted exactly to the local bounds.

    No collection, parenting, world transform, animation, or visibility changes
    are made. Objects are linked to bpy.context.collection. The canonical pose
    controls orientation, and the actual bounds are preserved vertex-for-vertex.
    color is sRGB in the range [0,1], converted to linear shader values here.
    """
    name=instance.get('instance_name',str(instance.get('instance_id','object')))
    category=instance.get('category','unknown')
    if len(bounds)!=6: raise ValueError('bounds must be xmin,xmax,ymin,ymax,zmin,zmax')
    lo=Vector((bounds[0],bounds[2],bounds[4]));hi=Vector((bounds[1],bounds[3],bounds[5]))
    size=hi-lo
    if any(not math.isfinite(v) or v<=0 for v in size):
        raise ValueError('Every bounding box extent must be positive and finite')
    canonical=instance.get('canonical_pose') or {}
    if 'up_vector' not in canonical or 'front_vector' not in canonical:
        raise ValueError('canonical_pose.up_vector and front_vector are required')
    up=Vector(canonical['up_vector']).normalized()
    front=Vector(canonical['front_vector'])
    front=(front-up*front.dot(up)).normalized()
    right=up.cross(front).normalized()
    orientation=Matrix((right,-front,up)).transposed()
    local_dimensions=[sum(abs(orientation[j][i])*size[j] for j in range(3)) for i in range(3)]
    m=_Mesh(_palette(instance,color))
    recipe=_recipe(m,category,name,local_dimensions)
    rotated=[orientation@Vector(v) for v in m.vertices]
    generated_lo=Vector(tuple(min(v[i] for v in rotated) for i in range(3)))
    generated_hi=Vector(tuple(max(v[i] for v in rotated) for i in range(3)))
    source_size=generated_hi-generated_lo
    fitted=[tuple(lo[i]+(v[i]-generated_lo[i])*size[i]/max(source_size[i],1e-9) for i in range(3)) for v in rotated]
    mesh=bpy.data.meshes.new('proxy_'+name)
    mesh.from_pydata(fitted,[],m.faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    for material in m.materials:mesh.materials.append(material)
    for polygon,index,smooth in zip(mesh.polygons,m.face_materials,m.smooth):
        polygon.material_index=index;polygon.use_smooth=smooth
    obj['geometry_source']='procedural_semantic_approximation_fitted_to_local_bounds'
    obj['semantic_recipe']=recipe
    obj['local_bounds']=list(map(float,bounds))
    obj['bounds_fit_max_error_m']=max(max(abs(min(v[i] for v in fitted)-lo[i]),
                                            abs(max(v[i] for v in fitted)-hi[i])) for i in range(3))
    return obj
