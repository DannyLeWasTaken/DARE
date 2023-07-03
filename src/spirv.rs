use phobos::ShaderCreateInfo;
use std::fs;
use std::io::Read;
use std::path::Path;

pub fn load_spirv_file(path: &Path) -> Vec<u32> {
    let mut f = std::fs::File::open(&path).expect("No SPIRV file found");
    let metadata = fs::metadata(&path).expect("Unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read_exact(&mut buffer).expect("Buffer overflow");
    let (_, binary, _) = unsafe { buffer.align_to::<u32>() };
    Vec::from(binary)
}

pub fn create_shader(path: &str, stage: phobos::vk::ShaderStageFlags) -> phobos::ShaderCreateInfo {
    let code = load_spirv_file(Path::new(path));
    ShaderCreateInfo::from_spirv(stage, code)
}
