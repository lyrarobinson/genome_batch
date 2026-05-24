"""
worm_body_physics.py — Cosserat rod body model for SOIL.

Rest-angle spring model with CPG sine wave. Parameters calibrated to give
chord/arc=0.63, ~1.5 wavelengths visible (biological crawling gait).

K_BEND=1.0, K_NORMAL=2.5, K_TANGENT=0.25, AMP=0.45rad, FREQ=0.5Hz.
Speed_override provides forward drive; _wave_amp scales amplitude+drive.

Authored departures:
  - CPG wave explicit (absent from B Full rate-coded model)
  - Speed via speed_override force along nose-neck heading
  - Connectome activations modulate wave amplitude (future work)
"""

import math
import numpy as np

N_PARTICLES  = 25
N_SEGMENTS   = 12
N_JOINTS     = N_PARTICLES - 1
N_BENDS      = N_PARTICLES - 2

BODY_LENGTH  = 20.0
REST_SEG_LEN = BODY_LENGTH / N_JOINTS  # ~0.4167 wu

K_STRETCH = 800
K_BEND    = 1.0
K_TANGENT = 0.25
K_NORMAL  = 2.5
DAMPING   = 2.0
K_MUSCLE  = 0.67
MASS      = 0.1

WAVE_FREQ     = 0.3
BASE_WAVE_AMP = 1.50

N_STEER_SEGS = 4
K_STEER      = 0.25

BODY_Y = 16.7


def seg_to_bend_indices(seg):
    return 2 * seg, 2 * seg + 1


class PhysicsWorm:

    def __init__(self, start_pos, start_heading=0.0,
                 k_stretch=K_STRETCH, k_bend=K_BEND,
                 k_tangent=K_TANGENT, k_normal=K_NORMAL,
                 k_muscle=K_MUSCLE, damping=DAMPING):

        self.k_stretch = k_stretch
        self.k_bend    = k_bend
        self.k_tangent = k_tangent
        self.k_normal  = k_normal
        self.k_muscle  = k_muscle
        self.damping   = damping

        self.reversing        = False
        self.reverse_time     = 0.0
        self.refractory_s     = 0.0
        self.quiescent        = False
        self.quiescent_ms     = 0.0
        self.eggs_accumulated = 0.0
        self.satiation        = 0.0
        self.eggs_laid        = 0
        self.pitch            = 0.0

        x0 = float(start_pos[0])
        z0 = float(start_pos[2]) if len(start_pos) > 2 else float(start_pos[1])
        heading = float(start_heading)

        dx = math.cos(heading) * REST_SEG_LEN
        dz = math.sin(heading) * REST_SEG_LEN
        self.pos_2d = np.zeros((N_PARTICLES, 2), dtype=np.float64)
        for i in range(N_PARTICLES):
            self.pos_2d[i, 0] = x0 - i * dx
            self.pos_2d[i, 1] = z0 - i * dz

        self.vel_2d = np.zeros((N_PARTICLES, 2), dtype=np.float64)
        self.rest_angles = np.zeros(N_BENDS, dtype=np.float64)
        self.muscle_dorsal  = np.zeros(N_SEGMENTS, dtype=np.float32)
        self.muscle_ventral = np.zeros(N_SEGMENTS, dtype=np.float32)

        self.x_min, self.x_max = 1.0, 959.0
        self.z_min, self.z_max = 1.0, 539.0

        self._body_phase    = math.pi / 2
        self._smooth_heading = heading
        # Undulation frequency: modulated by AVB interneuron activity.
        # AVB above baseline = forward locomotion state = higher frequency.
        # AVB below baseline = on food / slowing = lower frequency.
        # Range 0.3-0.6 Hz matches real C. elegans crawling (Fang-Yen et al. 2010).
        # Updated each step by worm_kinematic_sim_graded via _wave_freq setter.
        # Authored departure: AVB->B-motor frequency modulation proxied as
        # direct wave frequency parameter rather than synaptic weight change.
        self._wave_freq = 0.45  # default midpoint; updated from AVB each step

        self._pirouette_turn_t = 0.0
        self._pirouette_delta  = 0.0
        self._pirouette_heading = None
        self._pirouette_count  = 0
        self._food_world_x     = 0.0
        self._food_world_z     = 0.0

    @property
    def pos(self):
        return np.array([self.pos_2d[0, 0], BODY_Y, self.pos_2d[0, 1]])

    @pos.setter
    def pos(self, val):
        self.pos_2d[0, 0] = val[0]
        self.pos_2d[0, 1] = val[2] if len(val) > 2 else val[1]

    @property
    def heading(self):
        dx = self.pos_2d[0, 0] - self.pos_2d[1, 0]
        dz = self.pos_2d[0, 1] - self.pos_2d[1, 1]
        return math.atan2(dz, dx)

    @heading.setter
    def heading(self, val):
        self._rotate_body(val - self.heading)

    @property
    def heading_vec(self):
        h = self.heading
        return np.array([math.cos(h), 0.0, math.sin(h)])

    DT_BODY = 5e-4

    def step(self, dt, muscle_dorsal, muscle_ventral,
             speed_override=None, turn_signal=0.0, awc_signal=0.0):
        self._awc_signal = float(awc_signal)

        self.muscle_dorsal  = muscle_dorsal
        self.muscle_ventral = muscle_ventral

        if self.refractory_s > 0:
            self.refractory_s -= dt
        if self.reversing:
            self.reverse_time -= dt
            if self.reverse_time <= 0:
                self.reversing    = False
                self.reverse_time = 0.0
                # Refractory: shorter when leaving food (AWC active) so worm
                # can reorient quickly. Longer when on food so it stays in patch.
                # Biological basis: NSM serotonin prolongs forward runs on food;
                # absent from our model so proxied via AWC state.
                # Authored departure: AWC-state-dependent refractory period.
                _awc = getattr(self, '_awc_signal', 0.0)
                self.refractory_s = 1.0 if _awc > 0.01 else 3.0
                self._start_pirouette()

        if self._pirouette_turn_t > 0:
            # Pirouette: hold _pirouette_heading at target so forward drive
            # force pulls body toward new heading. Set once at start, held
            # until pirouette duration expires. Body physically swings toward
            # it over the pirouette duration -- no accumulation, no overshoot.
            # Biological basis: omega turn sets new heading via head swing;
            # body follows passively. Duration gives time to complete the turn.
            if self._pirouette_heading is None:
                # Cap pirouette to 90 degrees max per turn.
                # Prevents forward drive pointing backward during large reorientations.
                # Multiple pirouettes complete large heading corrections.
                # Biological basis: C. elegans omega turns are 30-130 deg
                # (Pierce-Shimomura et al. 1999); >180 deg turns don't occur.
                _capped_delta = float(np.clip(self._pirouette_delta,
                                              -math.pi*0.5, math.pi*0.5))
                self._pirouette_heading = self.heading + _capped_delta
            self._pirouette_turn_t -= dt
            if self._pirouette_turn_t <= 0:
                self._pirouette_turn_t = 0.0
                self._pirouette_delta  = 0.0
                self._pirouette_heading = None
                # Refractory resets after pirouette completes.
                _awc = getattr(self, '_awc_signal', 0.0)
                self.refractory_s = max(self.refractory_s,
                                        1.5 if _awc > 0.01 else 2.0)

        self._update_rest_angles(turn_signal)

        n_sub  = max(1, int(round(dt / self.DT_BODY)))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            forces = self._compute_forces(speed_override)
            self.vel_2d = (self.vel_2d + dt_sub * forces / MASS) / (1.0 + dt_sub * self.damping)
            self.pos_2d = self.pos_2d + dt_sub * self.vel_2d

        self._apply_boundaries()

        nv = self.vel_2d[0]; vmag = math.sqrt(nv[0]**2 + nv[1]**2)
        if vmag > 0.05:
            raw_h = math.atan2(nv[1], nv[0])
            alpha = min(dt / 2.0, 0.05)
            self._smooth_heading = math.atan2(
                (1-alpha)*math.sin(self._smooth_heading) + alpha*math.sin(raw_h),
                (1-alpha)*math.cos(self._smooth_heading) + alpha*math.cos(raw_h))

        return self.pos.copy(), self.heading_vec.copy(), self._body_points_3d()

    def _compute_forces(self, speed_override):
        forces = np.zeros((N_PARTICLES, 2), dtype=np.float64)

        # Stretch
        delta   = self.pos_2d[1:] - self.pos_2d[:-1]
        lengths = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
        unit    = delta / lengths[:, None]
        f       = self.k_stretch * (lengths - REST_SEG_LEN)[:, None] * unit
        forces[:-1] += f; forces[1:] -= f

        # Bend
        p0=self.pos_2d[:-2]; p1=self.pos_2d[1:-1]; p2=self.pos_2d[2:]
        v1=p1-p0; v2=p2-p1
        l1=np.maximum(np.linalg.norm(v1,axis=1),1e-10)
        l2=np.maximum(np.linalg.norm(v2,axis=1),1e-10)
        angles=np.arctan2(v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0],
                          v1[:,0]*v2[:,0]+v1[:,1]*v2[:,1])
        torque=-self.k_bend*(angles-self.rest_angles)
        perp1=np.stack([-v1[:,1],v1[:,0]],axis=1)/(l1**2)[:,None]
        perp2=np.stack([-v2[:,1],v2[:,0]],axis=1)/(l2**2)[:,None]
        forces[:-2]+=-torque[:,None]*perp1
        forces[2:]  += torque[:,None]*perp2
        forces[1:-1]+= torque[:,None]*(perp1-perp2)

        # Drag — scales with wave amplitude so reduced undulation = more drag.
        # At full _wave_amp: normal drag coefficients.
        # At reduced _wave_amp (on food/dense soil): drag increases, slowing worm.
        # Biological basis: undulation amplitude determines how effectively the
        # worm pushes against substrate; lower amplitude = less propulsion efficiency
        # = higher effective drag relative to forward force.
        # Authored departure: drag modulation proxies substrate coupling.
        _wamp = getattr(self, '_wave_amp', BASE_WAVE_AMP)
        _amp_ratio = _wamp / BASE_WAVE_AMP  # 0.2-1.0
        # Drag scales inversely with amplitude: less undulation = more drag
        _drag_scale = 1.0 / max(_amp_ratio, 0.2)
        tang=np.zeros((N_PARTICLES,2),dtype=np.float64)
        tang[1:-1]=self.pos_2d[2:]-self.pos_2d[:-2]
        tang[0]=self.pos_2d[1]-self.pos_2d[0]
        tang[-1]=self.pos_2d[-1]-self.pos_2d[-2]
        tl=np.maximum(np.linalg.norm(tang,axis=1,keepdims=True),1e-10); tang/=tl
        nv=np.stack([-tang[:,1],tang[:,0]],axis=1)
        vt=np.sum(self.vel_2d*tang,axis=1,keepdims=True)*tang
        vn=np.sum(self.vel_2d*nv,axis=1,keepdims=True)*nv
        forces-=_drag_scale*(self.k_tangent*vt+self.k_normal*vn)

        # Forward drive — suppressed during reversal
        _wave_amp = getattr(self, '_wave_amp', BASE_WAVE_AMP)
        if not self.reversing:
            spd = (_wave_amp / BASE_WAVE_AMP) * 2.2
            # Use pirouette heading if active, else nose-neck heading
            _pir_h = getattr(self, '_pirouette_heading', None)
            h = _pir_h if _pir_h is not None else self.heading
            fwd = np.array([math.cos(h), math.sin(h)])
            forces += spd * (self.damping + self.k_tangent) * fwd[None, :]

        return forces

    def _update_rest_angles(self, turn_signal=0.0):
        steer_w = np.array([max(0.0, 1.0-i/N_STEER_SEGS) for i in range(N_SEGMENTS)])
        steer_bias = K_STEER * float(turn_signal) * steer_w

        # Wave always travels head→tail. Reversal = sharp head bend, not backward wave.
        self._body_phase += 2.0 * math.pi * self._wave_freq * 5e-5

        wave_amp = getattr(self, '_wave_amp', BASE_WAVE_AMP)

        _turn_t = getattr(self, '_pirouette_turn_t', 0.0)
        pir_bias = 0.0
        # pir_bias removed: direct _pirouette_heading approach makes
        # body-wave bias redundant and causes backward drag at large deltas.

        for seg in range(N_SEGMENTS):
            s = seg / N_SEGMENTS
            if self.reversing:
                # Omega turn: head bends sharply, body goes passive
                if seg < 3:
                    angle = 2.5 * max(0.0, 1.0 - seg/3.0)
                else:
                    angle = 0.0
            else:
                # Damp wave during pirouette so heading force can reorient body.
                # Wave resumes at full amplitude after pirouette completes.
                _pir_damp = 0.2 if _turn_t > 0 else 1.0
                angle = wave_amp * _pir_damp * math.sin(self._body_phase - 2.0*math.pi*s) + steer_bias[seg]
            j0, j1 = seg_to_bend_indices(seg)
            if j0 < N_BENDS: self.rest_angles[j0] = angle
            if j1 < N_BENDS: self.rest_angles[j1] = angle

    def _start_pirouette(self):
        import random as _rng
        fmag = math.sqrt(self._food_world_x**2 + self._food_world_z**2)
        if fmag > 0.005:
            food_h = math.atan2(self._food_world_z, self._food_world_x)
            dh = food_h - self.heading
            while dh >  math.pi: dh -= 2*math.pi
            while dh < -math.pi: dh += 2*math.pi
            conf = min(1.0, fmag / 0.02)
            # Small corrective turn toward food -- real worms make 20-60 deg turns on gradient
            # Gain scales with heading error magnitude:
            # small errors (< 45 deg) -> 0.5x (gentle nudge)
            # large errors (> 135 deg) -> 0.9x (near-complete reorientation)
            # Biological basis: larger omega turns on steeper gradients
            # (Pierce-Shimomura et al. 1999 -- turn magnitude scales with
            # gradient strength; proxied here as heading error magnitude).
            _err_scale = 0.5 + 0.4 * (abs(dh) / math.pi)
            self._pirouette_delta = _rng.gauss(dh * conf * _err_scale, 0.2)
            self._food_world_x = 0.0
            self._food_world_z = 0.0
        else:
            self._pirouette_delta = _rng.gauss(0, 0.4)  # ~23 deg std -- real omega turn
        self._pirouette_count  += 1
        # Longer pirouette when leaving food: more time to complete large
        # heading corrections. On food, short pirouette to stay near patch.
        _awc = getattr(self, '_awc_signal', 0.0)
        self._pirouette_turn_t = 3.0 if _awc > 0.01 else 2.0

    def _apply_boundaries(self):
        x=self.pos_2d[:,0]; z=self.pos_2d[:,1]
        vx=self.vel_2d[:,0]; vz=self.vel_2d[:,1]
        mask=x<self.x_min; x[mask]=self.x_min; vx[mask]=abs(vx[mask])
        mask=x>self.x_max; x[mask]=self.x_max; vx[mask]=-abs(vx[mask])
        mask=z<self.z_min; z[mask]=self.z_min; vz[mask]=abs(vz[mask])
        mask=z>self.z_max; z[mask]=self.z_max; vz[mask]=-abs(vz[mask])

    def _body_points_3d(self):
        pts=np.zeros((N_PARTICLES,3),dtype=np.float32)
        pts[:,0]=self.pos_2d[:,0]; pts[:,1]=BODY_Y; pts[:,2]=self.pos_2d[:,1]
        return pts

    def get_body_points(self): return self._body_points_3d()

    def _rotate_body(self, delta):
        cd=math.cos(delta); sd=math.sin(delta); o=self.pos_2d[0].copy()
        for i in range(N_PARTICLES):
            p=self.pos_2d[i]-o
            self.pos_2d[i,0]=o[0]+p[0]*cd-p[1]*sd
            self.pos_2d[i,1]=o[1]+p[0]*sd+p[1]*cd

    def get_nose_tangent_perp(self):
        h=self._smooth_heading
        fwd=np.array([math.cos(h),math.sin(h)]); perp=np.array([-fwd[1],fwd[0]])
        nose=self.pos.copy()
        left =np.array([nose[0]+5.0*perp[0],BODY_Y,nose[2]+5.0*perp[1]])
        right=np.array([nose[0]-5.0*perp[0],BODY_Y,nose[2]-5.0*perp[1]])
        return nose, left, right

    def get_curvature(self):
        curvature=np.zeros(N_SEGMENTS,dtype=np.float32)
        for seg in range(N_SEGMENTS):
            j0,j1=seg_to_bend_indices(seg)
            vals=[]
            for j in (j0,j1):
                if j>=N_BENDS: continue
                p0=self.pos_2d[j];p1=self.pos_2d[j+1];p2=self.pos_2d[j+2]
                v1=p1-p0;v2=p2-p1
                l1=np.linalg.norm(v1);l2=np.linalg.norm(v2)
                if l1>1e-10 and l2>1e-10:
                    vals.append(math.atan2(v1[0]*v2[1]-v1[1]*v2[0],
                                           v1[0]*v2[0]+v1[1]*v2[1]))
            if vals: curvature[seg]=float(np.mean(vals))
        return curvature


if __name__ == '__main__':
    import time
    print("Testing PhysicsWorm (K_BEND=1.0, K_N=2.5, K_T=0.25)...")
    worm = PhysicsWorm(start_pos=np.array([100.0,BODY_Y,270.0]), start_heading=0.0)
    dt=0.05e-3; T=int(10.0/dt)
    muscle=np.zeros(N_SEGMENTS,dtype=np.float32)
    t0=time.time(); noses=[]
    for step in range(T):
        nose,_,body=worm.step(dt,muscle,muscle)
        if step%int(1.0/dt)==0:
            pts=worm.pos_2d
            arc=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
            chord=float(np.linalg.norm(pts[0]-pts[-1]))
            h=math.atan2(pts[0,1]-pts[-1,1],pts[0,0]-pts[-1,0])
            lat=math.sin(-h)*(pts[:,0]-pts[-1,0])+math.cos(-h)*(pts[:,1]-pts[-1,1])
            wl=len(np.where(np.diff(np.sign(lat)))[0])/2.0
            segs=np.linalg.norm(np.diff(pts,axis=0),axis=1)
            print(f"  t={step*dt:.0f}s nose=({nose[0]:.1f},{nose[2]:.1f})"
                  f" arc={arc:.2f} chord={chord:.2f} c/a={chord/arc:.2f}"
                  f" wl={wl:.1f} seg=[{segs.min():.3f},{segs.max():.3f}]")
            noses.append(nose.copy())
    wall=time.time()-t0; noses=np.array(noses)
    net=noses[-1]-noses[0]; drift=abs(net[2])/max(abs(net[0]),0.001)
    print(f"  Wall:{wall:.1f}s ({T*dt/wall:.1f}x) net=({net[0]:+.1f},{net[2]:+.1f}) drift={drift:.2f}")
