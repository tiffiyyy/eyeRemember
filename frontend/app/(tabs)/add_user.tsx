import React from 'react';
import {Alert,Button,StyleSheet,View} from 'react-native';

const ButtonBasics = ()=>{
    const onPress=()=>{
        Alert.alert("You are attempting to upload an image")
    };
    return(
        <View style={styles.container}>
            <View style={styles.buttonContainer}>
                <Button onPress={onPress} title="+Profile" color="#152084"/>
            </View>
        </View>
    );
};
const styles=StyleSheet.create({
    container:{
        flex:1,
        justifyContent:'center',
    },
    buttonContainer:{
        margin:20,
    },
});
export default ButtonBasics;