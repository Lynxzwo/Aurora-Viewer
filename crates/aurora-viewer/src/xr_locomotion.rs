//! Thumbstick walking and snap turning for the First Person VR view.
//!
//! This module only turns OpenXR thumbstick samples into movement *intentions*:
//! the normal walk controls and the VR rig yaw own the result, so the server
//! sees ordinary movement flags. Opt-in with `AURORA_VR_LOCOMOTION=1` (the
//! actions are created with the XR session, so it is read once at startup),
//! and the desktop switch starts off.
use aurora_render::{ControllerPoses, GripState, stereo_view::RigMode};
use glam::Vec2;

/// Radial dead zone of the walk stick; the walk magnitude is rescaled above it.
const DEAD_ZONE: f32 = 0.2;
/// Right-stick deflection that triggers one snap turn; it must be centred again to repeat.
const SNAP_THRESHOLD: f32 = 0.7;
const SNAP_ANGLE: f32 = std::f32::consts::FRAC_PI_6;

#[derive(Default)]
pub struct State {
    enabled: bool,
    open: bool,
    /// Per stick: centred since the last loss, focus or reference change.
    ready: [bool; 2],
    reference: Option<(u64, Option<u64>, u64, RigMode)>,
}

#[derive(Default, Debug)]
pub struct Motion {
    /// Forward/left, magnitude <= 1 after the radial dead zone.
    pub walk: Vec2,
    /// Yaw to add to the VR rig, in radians.
    pub snap: f32,
}

impl State {
    pub fn from_env() -> Option<Self> {
        (std::env::var("AURORA_VR_LOCOMOTION").as_deref() == Ok("1")).then(|| Self {
            open: true,
            ..Self::default()
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.reset();
        }
        self.enabled = enabled;
    }

    /// Forget the centred sticks and the reference: the next input must start from the centre.
    pub fn reset(&mut self) {
        self.ready = [false; 2];
        self.reference = None;
    }

    pub fn show(&mut self, ctx: &egui::Context, palette: &crate::theme::Palette) {
        let mut open = self.open;
        let mut enabled = self.enabled;
        crate::ui::widgets::Floater::new(
            "vr-locomotion",
            aurora_i18n::tr!("vr-walk-title"),
            egui::pos2(450.0, 75.0),
            egui::vec2(350.0, 180.0),
        )
        .show(ctx, palette, &mut open, |ui| {
            ui.label(aurora_i18n::tr!("vr-walk-help"));
            crate::ui::widgets::switch_labeled(ui, palette, &mut enabled, aurora_i18n::tr!("vr-walk-enable"));
        });
        self.open = open;
        // Closing the window disables the movement.
        self.set_enabled(open && enabled);
    }

    /// One frame of input. `None` (or a stale/foreign sample) releases everything.
    pub fn step(&mut self, sample: Option<ControllerPoses>, origin: Option<u64>, revision: u64) -> Motion {
        let Some(p) = sample.filter(|p| {
            self.enabled
                && p.matches(origin, revision)
                && p.mode == RigMode::FirstPerson
                && p.tracking_to_world.is_finite()
                && p.tracking_to_world.determinant().abs() > 1e-8
        }) else {
            self.reset();
            return Motion::default();
        };
        let reference = (p.generation, p.origin, p.revision, p.mode);
        if self.reference != Some(reference) {
            self.ready = [false; 2];
            self.reference = Some(reference);
        }
        let mut motion = Motion::default();
        for i in 0..2 {
            let axis = p.axes[i]
                .filter(|v| v.is_finite() && v.abs().max_element() <= 1.01)
                .filter(|_| matches!(p.hands[i], GripState::Tracked(v) if v.is_finite()));
            let Some(v) = axis else {
                self.ready[i] = false;
                continue;
            };
            if v.length() <= DEAD_ZONE {
                self.ready[i] = true;
            }
            if !self.ready[i] {
                continue;
            }
            if i == 0 {
                let length = v.length();
                let filtered = if length <= DEAD_ZONE {
                    Vec2::ZERO
                } else {
                    v / length * ((length - DEAD_ZONE) / (1.0 - DEAD_ZONE)).min(1.0)
                };
                // Stick (x right, y up) to avatar axes (forward, left).
                motion.walk = Vec2::new(filtered.y, -filtered.x);
            } else if v.x.abs() >= SNAP_THRESHOLD {
                motion.snap = -v.x.signum() * SNAP_ANGLE;
                self.ready[i] = false;
            }
        }
        motion
    }
}

impl Motion {
    /// The ordinary walk controls for this motion, or `None` when the stick is centred.
    pub fn input(&self) -> Option<crate::agent::MoveInput> {
        if self.walk == Vec2::ZERO {
            return None;
        }
        // Same eight directions as the VR body follow: the radial filter alone leaves
        // tiny cross-axis noise that must not become a full diagonal flag.
        let direction = ((self.walk.y.atan2(self.walk.x) / std::f32::consts::FRAC_PI_4).round() as i32).rem_euclid(8);
        Some(crate::agent::MoveInput {
            forward: matches!(direction, 0 | 1 | 7),
            back: matches!(direction, 3..=5),
            strafe_left: matches!(direction, 1..=3),
            strafe_right: matches!(direction, 5..=7),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};
    use std::time::{Duration, Instant};

    fn sample() -> ControllerPoses {
        ControllerPoses {
            hands: [GripState::Tracked(Vec3::ZERO); 2],
            axes: [Some(Vec2::ZERO); 2],
            tracking_to_world: Mat4::IDENTITY,
            buttons: [0; 2],
            generation: 1,
            origin: Some(7),
            revision: 2,
            mode: RigMode::FirstPerson,
            sampled: Instant::now(),
        }
    }
    fn enabled() -> State {
        State {
            enabled: true,
            ..State::default()
        }
    }

    #[test]
    fn xr_walk_directions_ignore_cross_axis_noise() {
        for (walk, expected) in [
            (Vec2::new(1.0, 0.02), [true, false, false, false]),
            (Vec2::new(-1.0, 0.02), [false, true, false, false]),
            (Vec2::new(0.02, 1.0), [false, false, true, false]),
            (Vec2::new(0.02, -1.0), [false, false, false, true]),
            (Vec2::ONE, [true, false, true, false]),
        ] {
            let input = Motion { walk, snap: 0.0 }.input().expect("direction");
            assert_eq!([input.forward, input.back, input.strafe_left, input.strafe_right], expected);
            assert!(!input.up && !input.down && !input.run, "walking never flies or runs");
        }
        assert!(Motion::default().input().is_none());
    }

    #[test]
    fn xr_walk_heading_and_control_flags_follow_head_after_snap() {
        use crate::{agent::AgentState, camera::vr_walk_yaw};
        use aurora_net::control;
        use glam::Quat;
        for rig in [0.0, std::f32::consts::FRAC_PI_2] {
            let head = 0.4;
            let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, Vec3::new(rig.cos(), rig.sin(), 0.0), Vec3::Z);
            let mut agent = AgentState::default();
            agent.on_server_update(Vec3::Z * 0.84, Vec3::ZERO, Vec3::ZERO, false, true);
            agent.yaw = vr_walk_yaw(view, Quat::from_rotation_y(head));
            assert!(
                agent
                    .forward()
                    .abs_diff_eq(Vec3::new((rig + head).cos(), (rig + head).sin(), 0.0), 1e-5)
            );
            for v in [Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y] {
                let input = Motion { walk: v, snap: 0.0 }.input().expect("direction");
                let flags = agent.control_flags(&input, true);
                let expected = if v.x > 0.0 {
                    control::AT_POS
                } else if v.x < 0.0 {
                    control::AT_NEG
                } else if v.y > 0.0 {
                    control::LEFT_POS
                } else {
                    control::LEFT_NEG
                };
                let linear = control::AT_POS | control::AT_NEG | control::LEFT_POS | control::LEFT_NEG;
                assert_eq!(flags & linear, expected);
            }
        }
    }

    #[test]
    fn xr_walk_switch_releases_disables_and_rearms() {
        use crate::agent::AgentState;
        use aurora_net::control;
        let mut s = State::default();
        let mut p = sample();
        assert!(!s.enabled(), "the switch starts off");
        assert_eq!(s.step(Some(p), p.origin, p.revision).walk, Vec2::ZERO);
        s.set_enabled(true);
        s.step(Some(p), p.origin, p.revision); // centred
        p.axes = [Some(Vec2::new(0.8, 0.8)), Some(Vec2::X)];
        let motion = s.step(Some(p), p.origin, p.revision);
        let input = motion.input().expect("direction");
        let flags = AgentState::default().control_flags(&input, true);
        assert_eq!(flags & (control::AT_POS | control::LEFT_NEG), control::AT_POS | control::LEFT_NEG);
        assert_ne!(motion.snap, 0.0);
        s.set_enabled(false);
        let stopped = s.step(Some(p), p.origin, p.revision);
        assert!(stopped.input().is_none());
        assert_eq!(stopped.snap, 0.0);
        s.set_enabled(true);
        assert!(
            s.step(Some(p), p.origin, p.revision).input().is_none(),
            "a held stick cannot resume"
        );
        p.axes = [Some(Vec2::ZERO); 2];
        s.step(Some(p), p.origin, p.revision);
        p.axes[0] = Some(Vec2::Y);
        assert!(s.step(Some(p), p.origin, p.revision).input().is_some());
        s.reset(); // an arrival can happen in one event batch
        assert!(s.step(Some(p), p.origin, p.revision).input().is_none());
        assert_eq!(s.step(None, p.origin, p.revision).walk, Vec2::ZERO);
    }

    #[test]
    fn xr_walk_radial_dead_zone_and_intensity() {
        let mut s = enabled();
        let mut p = sample();
        s.step(Some(p), Some(7), 2);
        for (raw, expected) in [
            (Vec2::new(0.1, 0.1), Vec2::ZERO),
            (Vec2::Y, Vec2::X),
            (Vec2::NEG_Y, Vec2::NEG_X),
            (Vec2::X, Vec2::NEG_Y),
            (Vec2::NEG_X, Vec2::Y),
            (Vec2::new(0.0, 0.6), Vec2::new(0.5, 0.0)),
        ] {
            p.axes[0] = Some(raw);
            assert!(s.step(Some(p), Some(7), 2).walk.abs_diff_eq(expected, 1e-6));
        }
        p.axes[0] = Some(Vec2::ONE);
        assert!((s.step(Some(p), Some(7), 2).walk.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn xr_walk_snap_requires_center_even_after_focus_reference_or_input_loss() {
        let mut s = enabled();
        let mut p = sample();
        p.axes[1] = Some(Vec2::X); // held on entry cannot turn
        assert_eq!(s.step(Some(p), Some(7), 2).snap, 0.0);
        p.axes[1] = Some(Vec2::ZERO);
        s.step(Some(p), Some(7), 2);
        p.axes[1] = Some(Vec2::X);
        assert_eq!(s.step(Some(p), Some(7), 2).snap, -SNAP_ANGLE);
        assert_eq!(s.step(Some(p), Some(7), 2).snap, 0.0);
        p.axes[1] = Some(Vec2::NEG_X); // reversal without centring cannot turn
        assert_eq!(s.step(Some(p), Some(7), 2).snap, 0.0);
        p.axes[1] = Some(Vec2::ZERO);
        s.step(Some(p), Some(7), 2);
        p.axes[1] = Some(Vec2::NEG_X);
        assert_eq!(s.step(Some(p), Some(7), 2).snap, SNAP_ANGLE);
        for loss in 0..5 {
            p.axes = [Some(Vec2::ZERO); 2];
            s.step(Some(p), Some(7), 2);
            let mut bad = p;
            match loss {
                0 => bad.sampled -= Duration::from_secs(1),
                1 => bad.generation += 1,
                2 => bad.axes = [Some(Vec2::splat(f32::NAN)); 2],
                3 => bad.hands = [GripState::Untracked; 2],
                _ => bad.mode = RigMode::Avatar,
            }
            assert_eq!(s.step(Some(bad), Some(7), 2).walk, Vec2::ZERO);
            p.axes = [Some(Vec2::ONE); 2];
            let m = s.step(Some(p), Some(7), 2);
            assert_eq!(m.walk, Vec2::ZERO, "sticks must be centred again after loss {loss}");
            assert_eq!(m.snap, 0.0);
        }
        s.step(None, Some(7), 2);
        assert_eq!(s.step(Some(p), Some(7), 2).walk, Vec2::ZERO);
    }

    #[test]
    fn xr_walk_invalid_controller_releases_only_its_intent() {
        let mut s = enabled();
        let mut p = sample();
        s.step(Some(p), Some(7), 2);
        p.hands[1] = GripState::Untracked;
        p.axes[0] = Some(Vec2::Y);
        assert_eq!(s.step(Some(p), Some(7), 2).walk, Vec2::X);
        p.hands[1] = GripState::Tracked(Vec3::ZERO);
        p.axes[1] = Some(Vec2::X);
        assert_eq!(s.step(Some(p), Some(7), 2).snap, 0.0);
        p.axes[1] = Some(Vec2::ZERO);
        s.step(Some(p), Some(7), 2);
        p.hands[0] = GripState::Inactive;
        p.axes[1] = Some(Vec2::X);
        let motion = s.step(Some(p), Some(7), 2);
        assert_eq!(motion.walk, Vec2::ZERO);
        assert_eq!(motion.snap, -SNAP_ANGLE);
    }
}
