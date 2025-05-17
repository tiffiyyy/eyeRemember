import React, {useState} from 'react';
import React, {Button, Image,View,StyleSheet} from 'react-native';
import * as ImagePicker from 'expo-image-picke';

export default function ImagePicker()
{
    cosnt[image,setImage]=useState<string|null>(null)
    const pickImage = async()=>{
        let result =await ImagePicker.launchImageLibraryAsync
        ({
            mediaTypes:['images'],
            allowsEditing: true, 
            aspect:[4,3],
            quality : 1,
        });
        console.log(result);
        if(!result.cancelled){
            setImage(result.assets[0].uri);
        }
    };
    return(
        <View style={StyleSheet.container}>
            <Button title = "Add Profile" onPress={pickImage}/>
            {image && <Image source={{uri:image}} style={styles.image}/>}
        </View>
    );
}
