package com.example.stage.Controller;

import com.example.stage.Model.Role;
import com.example.stage.Model.User;
import com.example.stage.service.EmailService;
import com.example.stage.service.UserService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/users")
public class UserController {

    private final UserService userService;
    private final ObjectMapper objectMapper = new ObjectMapper();
    @Autowired
    private EmailService emailService;
    @Autowired
    private RestTemplate restTemplate;
    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }

    // Création utilisateur
    @PostMapping(consumes = {"multipart/form-data"})
    public ResponseEntity<?> createUserWithPhoto(
            @RequestParam("file") MultipartFile file,
            @RequestParam("user") String userJson,
            @RequestParam("creatorId") String creatorId) {

        try {
            if (file.isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(Collections.singletonMap("messages", Collections.singletonList("Photo requise")));
            }

            byte[] fileBytes = file.getBytes();
            String photoHash = md5Hash(fileBytes);

            User user = objectMapper.readValue(userJson, User.class);
            if (user == null) {
                return ResponseEntity.badRequest()
                        .body(Collections.singletonMap("messages", Collections.singletonList("Données utilisateur invalides")));
            }

            if (user.getRole() == Role.EMPLOYEE) {
                user.setSupervisorId(creatorId);
            }

            user.setPhoto(fileBytes);
            user.setPhotoHash(photoHash);


            LinkedMultiValueMap<String, Object> map = new LinkedMultiValueMap<>();
            map.add("user", userJson);

            ByteArrayResource fileAsResource = new ByteArrayResource(fileBytes) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            };
            map.add("file", fileAsResource);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<LinkedMultiValueMap<String, Object>> requestEntity = new HttpEntity<>(map, headers);

            String flaskUrl = "http://localhost:5001/validate-user";
            ResponseEntity<Map> flaskResponse = restTemplate.postForEntity(flaskUrl, requestEntity, Map.class);

            if (!flaskResponse.getStatusCode().is2xxSuccessful()) {
                return ResponseEntity.badRequest().body(flaskResponse.getBody());
            }
            String hashedPassword = passwordEncoder.encode(user.getPassword());
            user.setPassword(hashedPassword);
            User createdUser = userService.createUser(user);
            return ResponseEntity.status(HttpStatus.CREATED).body(createdUser);

        } catch (HttpClientErrorException e) {
            String responseBody = e.getResponseBodyAsString();
            Map<String, Object> errorResponse;
            try {
                errorResponse = objectMapper.readValue(responseBody, Map.class);
            } catch (Exception ex) {
                errorResponse = Collections.singletonMap("messages", Collections.singletonList("Erreur validation IA inconnue."));
            }
            return ResponseEntity.badRequest().body(errorResponse);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Collections.singletonMap("messages", Collections.singletonList("Plusieurs champs sont manquants ou invalides.")));
        }
    }

    // Récupérer la photo
    @GetMapping("/photo/{userId}")
    public ResponseEntity<byte[]> getPhotoByUserId(@PathVariable String userId) {
        User user = userService.getUserById(userId);
        if (user != null && user.getPhoto() != null) {
            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_JPEG)
                    .body(user.getPhoto());
        }
        return ResponseEntity.notFound().build();
    }

    // Récupérer tous les utilisateurs
    @GetMapping
    public ResponseEntity<List<User>> getAll() {
        return ResponseEntity.ok(userService.getAllUsers());
    }

    // Récupérer un utilisateur par ID
    @GetMapping("/{id}")
    public ResponseEntity<?> getById(@PathVariable String id) {
        User user = userService.getUserById(id);
        return user != null ? ResponseEntity.ok(user) : ResponseEntity.notFound().build();
    }

    // Supprimer un utilisateur
    @DeleteMapping("/{id}")
    public ResponseEntity<?> delete(@PathVariable String id) {
        try {
            userService.deleteUser(id);
            return ResponseEntity.noContent().build();
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Erreur lors de la suppression");
        }
    }

    // Mettre à jour un utilisateur
    @PutMapping(value = "/{id}", consumes = {"multipart/form-data"})
    public ResponseEntity<?> updateUserWithPhoto(
            @PathVariable String id,
            @RequestParam(value = "file", required = false) MultipartFile file,
            @RequestParam("user") String userJson) {

        try {
            User existingUser = userService.getUserById(id);
            if (existingUser == null) {
                return ResponseEntity.notFound().build();
            }

            User userUpdates = objectMapper.readValue(userJson, User.class);

            if (file != null && !file.isEmpty()) {
                byte[] fileBytes = file.getBytes();
                userUpdates.setPhoto(fileBytes);
                userUpdates.setPhotoHash(md5Hash(fileBytes));
            } else {
                userUpdates.setPhoto(existingUser.getPhoto());
                userUpdates.setPhotoHash(existingUser.getPhotoHash());
            }

            userUpdates.setId(id);
            User updatedUser = userService.updateUser(userUpdates);
            return ResponseEntity.ok(updatedUser);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Erreur lors de la mise à jour");
        }
    }

    // Connexion utilisateur
    @PostMapping("/signin")
    public ResponseEntity<?> signIn(@RequestBody Map<String, String> credentials) {
        String email = credentials.get("email");
        String password = credentials.get("password");

        User user = userService.authenticate(email, password);
        return user != null ? ResponseEntity.ok(user)
                : ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Email ou mot de passe incorrect");
    }

    // Générer un hash MD5 pour la photo
    private String md5Hash(byte[] data) throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("MD5");
        byte[] digest = md.digest(data);
        StringBuilder sb = new StringBuilder();
        for (byte b : digest) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    // Obtenir les infos de profil
    @GetMapping("/profile/{id}")
    public ResponseEntity<?> getProfile(@PathVariable String id) {
        User user = userService.getUserById(id);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        Map<String, Object> response = new HashMap<>();
        response.put("id", user.getId());
        response.put("username", user.getUsername());
        response.put("email", user.getEmail());
        response.put("password", user.getPassword());
        response.put("role", user.getRole().name());
        response.put("phone", user.getPhone());
        response.put("address", user.getAddress());
        response.put("photoUrl", "http://localhost:8081/users/photo/" + user.getId());

        return ResponseEntity.ok(response);
    }

    // Obtenir l'utilisateur courant par email
    @GetMapping("/me")
    public ResponseEntity<?> getCurrentUser(@RequestParam("email") String email) {
        User user = userService.getUserByEmail(email);
        if (user == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Utilisateur non trouvé");
        }
        Map<String, Object> response = new HashMap<>();
        response.put("id", user.getId());
        response.put("username", user.getUsername());
        response.put("email", user.getEmail());
        response.put("role", user.getRole().name());
        response.put("phone", user.getPhone());
        response.put("address", user.getAddress());
        return ResponseEntity.ok(response);
    }

    // 1. Envoi de code reset par email
    @PostMapping("/forgot-password")
    public ResponseEntity<?> forgotPassword(@RequestBody Map<String, String> payload) {
        String email = payload.get("email");
        User user = userService.getUserByEmail(email);
        if (user == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Collections.singletonMap("message", "Email non trouvé"));
        }
        // Générer un code à 6 chiffres
        String code = String.valueOf(new Random().nextInt(900000) + 100000);
        user.setResetCode(code);
        userService.updateUser(user);

        emailService.sendEmail(email, "Réinitialisation de mot de passe",
                "Votre code de réinitialisation est : " + code);

        return ResponseEntity.ok(Collections.singletonMap("message", "Code envoyé par email"));
    }

    // 2. Réinitialiser le mot de passe avec code
    @PostMapping("/reset-password")
    public ResponseEntity<?> resetPassword(@RequestBody Map<String, String> payload) {
        String email = payload.get("email");
        String code = payload.get("code");
        String newPassword = payload.get("newPassword");

        User user = userService.getUserByEmail(email);
        if (user == null || user.getResetCode() == null || !user.getResetCode().equals(code)) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Collections.singletonMap("message", "Code invalide ou email incorrect"));
        }

        user.setPassword(passwordEncoder.encode(newPassword));
        user.setResetCode(null);
        userService.updateUser(user);

        return ResponseEntity.ok(Collections.singletonMap("message", "Mot de passe réinitialisé avec succès"));
    }

}
