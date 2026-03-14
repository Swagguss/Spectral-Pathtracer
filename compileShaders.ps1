$ErrorActionPreference = "Stop"

try {
    glslangValidator -V Shared/fullscreen.vert -o Shared/fullscreen.vert.spv
    glslangValidator -V Shared/present.frag -o Shared/present.frag.spv
    glslangValidator -V Shared/wavefront_rt.comp -o Shared/wavefront_rt.comp.spv
	glslangValidator -V Shared/denoise.comp -o Shared/denoise.comp.spv
    Write-Host "Shader compilation succeeded."
}
catch {
    Write-Host "Shader compilation failed:"
    Write-Host $_
}
finally {
    Read-Host "Press Enter to close"
}