'use strict';


module.exports = (sequelize, DataTypes) => {
  const Shortlist = sequelize.define('Shortlist', {

  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'ShortlistedModels'
  });

  Shortlist.associate = function(models) {
  	models.Shortlist.belongsTo(models.RunDetail, {
      foreignKey:'RunId'
    });
    models.Shortlist.belongsTo(models.ModelRunDetail, {
      foreignKey:'ModelId'
    });
  };
  return Shortlist;
};
