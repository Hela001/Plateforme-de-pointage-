package com.example.stage.service;


import com.example.stage.Model.User;
import com.example.stage.Repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
@Service
public class UserServiceImpl implements UserService {

    private final UserRepository repo;
    private PasswordEncoder passwordEncoder;


    public UserServiceImpl(UserRepository repo, PasswordEncoder passwordEncoder) {
        this.repo = repo;
        this.passwordEncoder = passwordEncoder;
    }

    @Override
    public User createUser(User user) {
        System.out.println("Creating user: " + user);
        return repo.save(user);
    }

    @Override
    public List<User> getAllUsers() {
        return repo.findAll();
    }

    @Override
    public User getUserById(String id) {
        return repo.findById(id).orElse(null);
    }

    @Override
    public void deleteUser(String id) {
        repo.deleteById(id);
    }

    @Override
    public User updateUser(User user) {
        if (user.getId() == null || !repo.existsById(user.getId())) {
            throw new IllegalArgumentException("User with ID " + user.getId() + " not found");
        }
        return repo.save(user);
    }


    @Override
    public User authenticate(String email, String password) {
        User user = repo.findByEmail(email);
        if (user != null && passwordEncoder.matches(password, user.getPassword())) {
            return user;
        }
        return null;
    }

    public User getUserByEmail(String email) {
        return repo.findByEmail(email);
    }

}
