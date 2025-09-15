package com.example.stage.service;
import com.example.stage.Model.User;

import java.util.List;

public interface UserService {
    User createUser(User user);
    List<User> getAllUsers();
    User getUserById(String id);
    void deleteUser(String id);
    User updateUser(User user);
    User getUserByEmail(String email);
    User authenticate(String email, String password);

}
