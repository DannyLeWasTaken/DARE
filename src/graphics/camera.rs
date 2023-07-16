//! Camera class to fly around the scene
use glam;

/// Represents a camera class that graphics will use
pub struct Camera {
    pub position: glam::Vec3,
    pub direction: glam::Vec3,
    pub world_up: glam::Vec3,
    pub up: glam::Vec3,
    pub front: glam::Vec3,
    pub right: glam::Vec3,

    pub view: glam::Mat4,
    pub speed: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Camera {
    pub fn new() -> Self {
        let position = glam::Vec3::new(0.0, 0.0, 0.0);
        let front = glam::Vec3::new(0.0, 0.0, -1.0);
        let up = glam::Vec3::new(0.0, 1.0, 0.0);

        Self {
            position,
            direction: glam::Vec3::new(0.0, 0.0, -1.0),
            front,
            right: glam::Vec3::normalize(glam::Vec3::cross(up, front)),
            world_up: up,
            up,
            view: glam::Mat4::look_at_rh(position, position + front, up),
            speed: 16.0,
            pitch: 0.0,
            yaw: 90.0,
        }
    }

    /// Update camera view
    pub fn update_camera(&mut self) -> glam::Mat4 {
        //self.up = glam::Vec3::normalize(glam::Mat4::cross(self.get_right()));
        let mut front: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 0.0);
        front.x = f32::cos(f32::to_radians(self.yaw)) * f32::cos(f32::to_radians(self.pitch));
        front.y = f32::sin(f32::to_radians(self.pitch));
        front.z = f32::sin(f32::to_radians(self.yaw)) * f32::cos(f32::to_radians(self.pitch));
        self.front = glam::Vec3::normalize(front);
        self.right = glam::Vec3::normalize(glam::Vec3::cross(self.front, self.world_up));
        self.up = glam::Vec3::normalize(glam::Vec3::cross(self.right, self.front));

        let look_at: glam::Mat4 =
            glam::Mat4::look_at_rh(self.position, self.position + self.front, self.world_up);
        self.view = look_at;

        look_at
    }
}
