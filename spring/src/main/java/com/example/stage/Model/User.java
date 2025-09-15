package com.example.stage.Model;

import org.bson.types.Binary;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "user")
public class User {

    @Id
    private String id;
    private String username;
    private String email;
    private String password;
    private Role role;
    private byte[]  photo;
    private String supervisorId;
    private String photoHash;
    private String phone;
    private String address;
    private String resetCode;
    public String getResetCode() { return resetCode; }
    public void setResetCode(String resetCode) { this.resetCode = resetCode; }

    public User() {}

    public User(String id, String username, String email, String password, Role role, byte[]  photo, String supervisorId, String photoHash, String phone, String address) {
        this.id = id;
        this.username = username;
        this.email = email;
        this.password = password;
        this.role = role;
        this.photo = photo;
        this.supervisorId = supervisorId;
        this.photoHash = photoHash;
        this.phone = phone;
        this.address = address;
    }
    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getId() {
        return id;
    }
    public void setId(String id) {
        this.id = id;
    }
    public String getUsername() {
        return username;
    }
    public void setUsername(String username) {
        this.username = username;
    }
    public String getEmail() {
        return email;
    }
    public void setEmail(String email) {
        this.email = email;
    }
    public String getPassword() {
        return password;
    }
    public void setPassword(String password) {
        this.password = password;
    }
    public Role getRole() {
        return role;
    }
    public void setRole(Role role) {
        this.role = role;
    }
    public byte[]  getPhoto() {
        return photo;
    }
    public void setPhoto(byte[]  photo) {
        this.photo = photo;
    }

    public String getSupervisorId() {
        return supervisorId;
    }

    public void setSupervisorId(String supervisorId) {
        this.supervisorId = supervisorId;
    }
    public String getPhotoHash() {
        return photoHash;
    }
    public void setPhotoHash(String photoHash) {
        this.photoHash = photoHash;
    }
}
