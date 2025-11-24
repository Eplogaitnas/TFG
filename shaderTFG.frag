#version 330 core
in vec3 fragNormal;
in vec3 fragPos;

out vec4 FragColor;


uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;

void main()
{
    // Sombreado Phong básico
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPos);

    // Difuso
    float diff = max(dot(norm, lightDir), 0.30);

    // Especular
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.30), 32);

    float ambient = 0.3; // Menos luz ambiental
    vec3 lightColor = vec3(1.0, 1.0, 1.0); // Luz blanca
    float diffuseFactor = 2.5; // Más contraste difuso
    float specularFactor = 1.2; // Más brillo especular
    vec3 color = (ambient + diffuseFactor * diff + specularFactor * spec) * objectColor * lightColor;
    FragColor = vec4(color, 1.0);
}